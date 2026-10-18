"""
Training Pipeline for Visual Question Answering (VQA).

This script:
1. Loads questions, answers.json, vocab.json, and pre-extracted image features (features.pt).
2. Filters out samples whose answers are not in answers.json and reports how many were skipped.
3. Splits train and validation sets strictly by IMAGE (no image appears in both train and val).
4. Trains VQAModel using CrossEntropyLoss and Adam optimizer (lr=1e-3, batch size=64).
5. Reports train loss and validation accuracy every epoch.
6. Saves checkpoints every epoch to CHECKPOINT_DIR and tracks the best model by validation accuracy.
7. Applies early stopping with patience 3.
8. Supports --fake flag for end-to-end dry-run testing before real features exist.
9. Supports --limit N flag to train on a small subset of samples.

Usage:
    # Run fake dry-run test with 200 samples:
    python train.py --fake --limit 200

    # Real training:
    python train.py --questions data/questions.json \
                    --features features.pt \
                    --vocab vocab.json \
                    --answers answers.json \
                    --checkpoint_dir checkpoints
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from text_encoder.tokenizer import encode_question
from models.vqa_model import VQAModel


# ==============================================================================
# CONFIGURATION BLOCK
# Default paths and hyperparameters.
# These can all be overridden from the command line.
# Note: On Google Colab, set CHECKPOINT_DIR and data paths to Google Drive!
# ==============================================================================
DEFAULT_CONFIG = {
    "questions_path": "data/questions.json",
    "annotations_path": None,
    "vocab_path": "vocab.json",
    "answers_path": "answers.json",
    "features_path": "features.pt",
    "checkpoint_dir": "checkpoints",
    "val_split": 0.2,           # 20% of unique images held out for validation
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 15,
    "early_stopping_patience": 3,
    "dropout": 0.5,
    "max_len": 20,              # Max token sequence length
    "seed": 42,
}


# ==============================================================================
# DATASET IMPLEMENTATION
# ==============================================================================
class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA.
    Pairs pre-extracted ResNet-50 image features [2048] with tokenized questions [20]
    and integer class labels [0..2999].
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        features_dict: Dict[str, torch.Tensor],
        vocab: Dict[str, int],
        max_len: int = 20
    ):
        """
        Args:
            samples: List of dicts, each with keys 'image', 'question', 'label'.
            features_dict: Dict mapping image identifiers to [2048] feature tensors.
            vocab: Mapping of word string -> token ID.
            max_len: Fixed sequence length for question tokens.
        """
        self.samples = samples
        self.features_dict = features_dict
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def _lookup_feature(self, img_key: Any) -> torch.Tensor:
        """
        Retrieves feature vector from features_dict, handling common key mismatches
        (e.g., int ID vs string, with or without .jpg extension).
        """
        if img_key in self.features_dict:
            feat = self.features_dict[img_key]
        elif str(img_key) in self.features_dict:
            feat = self.features_dict[str(img_key)]
        elif f"{img_key}.jpg" in self.features_dict:
            feat = self.features_dict[f"{img_key}.jpg"]
        elif os.path.basename(str(img_key)) in self.features_dict:
            feat = self.features_dict[os.path.basename(str(img_key))]
        else:
            raise KeyError(f"Image key '{img_key}' was not found in features.pt.")

        # Ensure feature is a FloatTensor of shape [2048]
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        elif feat.dtype != torch.float32:
            feat = feat.to(torch.float32)

        return feat

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self.samples[idx]

        # 1. Retrieve cached image feature [2048]
        image_feature = self._lookup_feature(item["image"])

        # 2. Tokenize and pad/truncate question -> LongTensor [20]
        question_ids = encode_question(item["question"], self.vocab, max_len=self.max_len)

        # 3. Target class index (integer in 0..2999)
        label = item["label"]

        return image_feature, question_ids, label


# ==============================================================================
# DATA LOADING & FILTERING
# ==============================================================================
def load_raw_questions(questions_path: str, annotations_path: str = None) -> List[Dict[str, Any]]:
    """
    Loads raw questions JSON file and returns standardized list of sample dicts.
    """
    if not os.path.isfile(questions_path):
        raise FileNotFoundError(f"Questions file not found at: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Questions JSON must be a list or a dict containing 'questions'.")

    # If separate annotations file exists (e.g. VQA v2)
    if annotations_path:
        with open(annotations_path, "r", encoding="utf-8") as f:
            anno_data = json.load(f)
        anno_list = anno_data.get("annotations", anno_data if isinstance(anno_data, list) else [])
        anno_map = {a["question_id"]: a for a in anno_list if "question_id" in a}

        merged = []
        for q in items:
            combined = {**q, **anno_map.get(q.get("question_id"), {})}
            merged.append(combined)
        items = merged

    standardized = []
    for item in items:
        # Resolve image identifier
        img = item.get("image") or item.get("image_name") or item.get("image_id") or ""
        # Resolve question text
        ques = item.get("question") or item.get("question_str") or item.get("text") or ""
        # Resolve answer text
        ans = ""
        if "answer" in item and isinstance(item["answer"], str):
            ans = item["answer"]
        elif "multiple_choice_answer" in item and isinstance(item["multiple_choice_answer"], str):
            ans = item["multiple_choice_answer"]
        elif "answers" in item and isinstance(item["answers"], list) and len(item["answers"]) > 0:
            first_ans = item["answers"][0]
            ans = first_ans.get("answer", "") if isinstance(first_ans, dict) else str(first_ans)

        standardized.append({
            "image": img,
            "question": str(ques).strip(),
            "answer": str(ans).strip().lower()
        })

    return standardized


def split_by_image(
    samples: List[Dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """
    Splits samples into train and validation sets strictly by image.
    Guarantees that NO image in the training set appears in the validation set.

    Args:
        samples: Filtered list of samples.
        val_split: Fraction of unique images reserved for validation.
        seed: Random seed for deterministic reproducibility.

    Returns:
        train_samples, val_samples, num_train_images, num_val_images
    """
    # 1. Group question samples by image
    image_groups = defaultdict(list)
    for sample in samples:
        image_groups[sample["image"]].append(sample)

    unique_images = list(image_groups.keys())
    unique_images.sort()  # Sort first for cross-platform deterministic order

    # 2. Shuffle unique images deterministically
    rng = random.Random(seed)
    rng.shuffle(unique_images)

    total_images = len(unique_images)
    # Reserve at least 1 image for val if more than 1 image exists
    num_val_images = max(1, int(total_images * val_split)) if total_images > 1 else 0
    num_train_images = total_images - num_val_images

    train_image_keys = set(unique_images[:num_train_images])
    val_image_keys = set(unique_images[num_train_images:])

    # Strict assertion: no image overlap
    assert len(train_image_keys.intersection(val_image_keys)) == 0, "Train and Val images must not overlap!"

    # 3. Assemble sample lists
    train_samples = [s for img in train_image_keys for s in image_groups[img]]
    val_samples = [s for img in val_image_keys for s in image_groups[img]]

    return train_samples, val_samples, num_train_images, num_val_images


# ==============================================================================
# SYNTHETIC DATA GENERATOR (--fake flag)
# ==============================================================================
def generate_fake_data(
    num_samples: int = 200,
    vocab_path: str = None,
    answers_path: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, int], List[str]]:
    """
    Generates synthetic questions, answers, vocabulary, and image features.
    Enables testing the entire pipeline before Person A provides features.pt.
    """
    print("[FAKE MODE] Generating synthetic questions, answers, and image features...")

    # 1. Load or synthesize answers (3000 classes)
    if answers_path and os.path.isfile(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
    else:
        answers = [f"answer_{i}" for i in range(3000)]

    # 2. Load or synthesize vocabulary
    if vocab_path and os.path.isfile(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    else:
        vocab = {"<pad>": 0, "<unk>": 1}
        for i in range(1000):
            vocab[f"word_{i}"] = i + 2

    # 3. Synthesize 50 unique images with random 2048-dim feature vectors
    num_images = 50
    features_dict = {}
    image_names = [f"fake_img_{i}.jpg" for i in range(num_images)]
    for img_name in image_names:
        features_dict[img_name] = torch.randn(2048, dtype=torch.float32)

    # 4. Synthesize questions distributed across these images
    raw_samples = []
    vocab_words = [w for w in vocab.keys() if w not in ("<pad>", "<unk>")]
    rng = random.Random(42)

    for i in range(num_samples):
        img_name = image_names[i % num_images]
        # Pick 3 to 7 random words from vocab
        q_len = rng.randint(3, 7)
        q_words = [rng.choice(vocab_words) for _ in range(q_len)]
        question_text = " ".join(q_words) + " ?"

        # Pick an answer: 90% in top answers, 10% out-of-vocab to test skipping
        if rng.random() < 0.90:
            answer_text = rng.choice(answers[:500])
        else:
            answer_text = f"rare_unseen_answer_{i}"

        raw_samples.append({
            "image": img_name,
            "question": question_text,
            "answer": answer_text
        })

    return raw_samples, features_dict, vocab, answers


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_model(config: Dict[str, Any]):
    print("=" * 70)
    print("VQA MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Select compute device: CUDA GPU -> Apple MPS -> CPU
    if config["device"] and config["device"] != "auto":
        device = torch.device(config["device"])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # Set seeds for reproducibility
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    if config["fake"]:
        target_samples = config["limit"] if config["limit"] else 200
        raw_samples, features_dict, vocab, answers = generate_fake_data(
            num_samples=target_samples,
            vocab_path=config["vocab_path"],
            answers_path=config["answers_path"]
        )
    else:
        print(f"Loading vocabulary from: {config['vocab_path']}")
        with open(config["vocab_path"], "r", encoding="utf-8") as f:
            vocab = json.load(f)

        print(f"Loading top answers from: {config['answers_path']}")
        with open(config["answers_path"], "r", encoding="utf-8") as f:
            answers = json.load(f)

        print(f"Loading image features from: {config['features_path']}")
        features_dict = torch.load(config["features_path"], map_location="cpu")
        if not isinstance(features_dict, dict):
            raise TypeError("Expected features.pt to contain a dict of {image_name: tensor}.")

        print(f"Loading questions from: {config['questions_path']}")
        raw_samples = load_raw_questions(config["questions_path"], config["annotations_path"])

    print(f"Loaded {len(raw_samples):,} raw question items.")
    print(f"Vocabulary size: {len(vocab):,} tokens.")
    print(f"Number of answer classes: {len(answers):,} classes.")

    # -------------------------------------------------------------------------
    # 2. FILTER SAMPLES & MAP ANSWERS TO CLASS IDs
    # -------------------------------------------------------------------------
    ans_to_id = {ans: idx for idx, ans in enumerate(answers)}
    valid_samples = []
    skipped_count = 0

    for sample in raw_samples:
        ans = sample["answer"]
        if ans in ans_to_id:
            valid_samples.append({
                "image": sample["image"],
                "question": sample["question"],
                "answer": ans,
                "label": ans_to_id[ans]
            })
        else:
            skipped_count += 1

    print(f"Filtered samples:")
    print(f"  - Skipped: {skipped_count:,} samples (answer not in top {len(answers)} answers).")
    print(f"  - Retained: {len(valid_samples):,} valid samples.")

    if not valid_samples:
        raise RuntimeError("No valid samples remained after filtering! Check answers.json.")

    # Apply --limit N if requested
    if config["limit"] is not None and config["limit"] > 0:
        valid_samples = valid_samples[:config["limit"]]
        print(f"Applied --limit: Training on the first {len(valid_samples)} samples.")

    # -------------------------------------------------------------------------
    # 3. SPLIT TRAIN / VAL BY IMAGE
    # -------------------------------------------------------------------------
    train_samples, val_samples, n_train_imgs, n_val_imgs = split_by_image(
        valid_samples,
        val_split=config["val_split"],
        seed=seed
    )

    print("-" * 70)
    print("Train/Val Split Summary (Strict Image-Level Disjoint Split):")
    print(f"  Train: {len(train_samples):,} samples across {n_train_imgs:,} unique images")
    print(f"  Val:   {len(val_samples):,} samples across {n_val_imgs:,} unique images")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # 4. CREATE DATA LOADERS
    # -------------------------------------------------------------------------
    train_dataset = VQADataset(train_samples, features_dict, vocab, max_len=config["max_len"])
    val_dataset = VQADataset(val_samples, features_dict, vocab, max_len=config["max_len"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False
    )

    # -------------------------------------------------------------------------
    # 5. INITIALIZE MODEL, LOSS & OPTIMIZER
    # -------------------------------------------------------------------------
    model = VQAModel(
        vocab_size=len(vocab),
        image_dim=2048,
        text_dim=1024,
        num_classes=len(answers),
        dropout_prob=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Prepare checkpoint directory
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(checkpoint_dir)}")

    # -------------------------------------------------------------------------
    # 6. TRAINING & VALIDATION LOOP
    # -------------------------------------------------------------------------
    best_val_acc = -1.0
    patience = config["early_stopping_patience"]
    patience_counter = 0

    print("Starting training loop (up to {} epochs)...".format(config["num_epochs"]))
    print("=" * 70)

    for epoch in range(1, config["num_epochs"] + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for img_feats, q_tokens, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            img_feats = img_feats.to(device)
            q_tokens = q_tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(img_feats, q_tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        correct_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for img_feats, q_tokens, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  "):
                img_feats = img_feats.to(device)
                q_tokens = q_tokens.to(device)
                labels = labels.to(device)

                logits = model(img_feats, q_tokens)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()
                val_batches += 1

                # Calculate accuracy: predicted class is argmax over the 3000 scores
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_accuracy = (correct_predictions / max(1, total_val_samples)) * 100.0

        # Print epoch metrics
        print(
            f"Epoch {epoch:02d}/{config['num_epochs']:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% ({correct_predictions}/{total_val_samples})"
        )

        # Save checkpoint every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_accuracy,
            "vocab_size": len(vocab),
            "num_classes": len(answers),
        }
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_ckpt_path)

        # Keep the best checkpoint by validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_ckpt_path)
            print(f"  --> Saved new best checkpoint: {best_ckpt_path} (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  --> Validation accuracy did not improve. (Patience: {patience_counter}/{patience})")

        # Early stopping condition
        if patience_counter >= patience:
            print("-" * 70)
            print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
            break

    print("=" * 70)
    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train VQA Language and Classifier Pipeline.")

    # Paths (configurable via CLI or top config block)
    parser.add_argument("--questions", type=str, default=DEFAULT_CONFIG["questions_path"],
                        help="Path to questions JSON file.")
    parser.add_argument("--annotations", type=str, default=DEFAULT_CONFIG["annotations_path"],
                        help="Optional path to separate annotations JSON file.")
    parser.add_argument("--vocab", type=str, default=DEFAULT_CONFIG["vocab_path"],
                        help="Path to vocab.json.")
    parser.add_argument("--answers", type=str, default=DEFAULT_CONFIG["answers_path"],
                        help="Path to answers.json.")
    parser.add_argument("--features", type=str, default=DEFAULT_CONFIG["features_path"],
                        help="Path to features.pt file containing image feature dict.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CONFIG["checkpoint_dir"],
                        help="Directory to save model checkpoints (e.g. Google Drive path).")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Mini-batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Adam learning rate (default: 1e-3).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Maximum training epochs (default: 15).")
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["early_stopping_patience"],
                        help="Early stopping patience in epochs (default: 3).")
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"],
                        help="Dropout probability (default: 0.5).")
    parser.add_argument("--val_split", type=float, default=DEFAULT_CONFIG["val_split"],
                        help="Fraction of unique images held out for validation (default: 0.2).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'cuda', 'mps', 'cpu', or 'auto'.")

    # Testing & Debug flags
    parser.add_argument("--fake", action="store_true",
                        help="Use synthetic questions and random features for dry-run testing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit training to only N samples for quick sanity check.")

    args = parser.parse_args()

    config = {
        "questions_path": args.questions,
        "annotations_path": args.annotations,
        "vocab_path": args.vocab,
        "answers_path": args.answers,
        "features_path": args.features,
        "checkpoint_dir": args.checkpoint_dir,
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "dropout": args.dropout,
        "max_len": DEFAULT_CONFIG["max_len"],
        "seed": DEFAULT_CONFIG["seed"],
        "fake": args.fake,
        "limit": args.limit,
        "device": args.device,
    }

    train_model(config)


if __name__ == "__main__":
    main()
