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
Supports official VQA v2 dataset formats:
- Questions file: {"questions": [{"question_id", "image_id", "question"}, ...]}
- Annotations file: {"annotations": [{"question_id", "image_id", "multiple_choice_answer",
                                      "answers": [10 x {"answer"}], "answer_type", "question_type"}, ...]}
- Image filenames: COCO_train2014_000000XXXXXX.jpg (zero-padded to 12 digits).
- Training label: multiple_choice_answer mapped to top 3000 classes.
- Validation evaluation: Official VQA accuracy min(matches / 3, 1.0) over 10 answers,
                         reported overall and broken down per answer_type (yes/no, number, other).
- Subsampling: --limit_images N (keeps all questions for N unique images).
- Validation split: Either official val2014 (--val_questions) OR a held-out slice of train split by image.
- Testing flags: --fake (synthetic dry-run) and --limit N (sample limit).
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from typing import Dict, List, Tuple, Any, Optional

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
# In Google Colab, set paths to your Google Drive location:
# "MyDrive/VQA Project/dataset/..."
# ==============================================================================
DEFAULT_CONFIG = {
    "questions_path": "data/questions.json",
    "annotations_path": None,
    # Train files
    "questions_path": "data/v2_OpenEnded_mscoco_train2014_questions.json",
    "annotations_path": "data/v2_mscoco_train2014_annotations.json",
    "features_path": "features.pt",
    
    # Validation files (optional: if omitted, splits train set by image)
    "val_questions_path": None,
    "val_annotations_path": None,
    "val_features_path": None,

    # Shared vocabulary and answers list
    "vocab_path": "vocab.json",
    "answers_path": "answers.json",
    "features_path": "features.pt",
    
    # Checkpoints & Hyperparameters
    "checkpoint_dir": "checkpoints",
    "val_split": 0.2,           # 20% of unique images held out for validation
    "val_split": 0.2,           # 20% of images held out if separate val set not provided
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 15,
    "early_stopping_patience": 3,
    "dropout": 0.5,
    "max_len": 20,              # Max token sequence length
    "seed": 42,
    "limit_images": None,       # Subsample to N unique images (all their questions)
    "limit": None,              # Subsample to N total questions
}


# ==============================================================================
# VQA ACCURACY METRIC
# ==============================================================================
def compute_vqa_accuracy(predicted_answer: str, annotator_answers: List[str]) -> float:
    """
    Computes the official VQA competition accuracy metric for a single sample:
        accuracy = min(matches / 3.0, 1.0)
    where matches is the count of how many of the 10 human annotators gave the predicted answer.
    """
    matches = sum(1 for ans in annotator_answers if ans == predicted_answer)
    return min(matches / 3.0, 1.0)


# ==============================================================================
# DATASET IMPLEMENTATION
# ==============================================================================
class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA.
    Pairs pre-extracted ResNet-50 image features [2048] with tokenized questions [20]
    and integer class labels [0..2999].
    and integer class labels [0..2999], returning sample index for metric evaluation.
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
        Retrieves feature vector from features_dict, handling common key variations:
        - Exact string key (e.g. 'COCO_train2014_000000000025.jpg')
        - Zero-padded 12-digit int or digit string
        - Basename
        """
        # 1. Direct match
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
            # 2. Try zero-padded 12-digit COCO patterns if key is numeric
            int_val = None
            if isinstance(img_key, int):
                int_val = img_key
            elif isinstance(img_key, str):
                digits = "".join(filter(str.isdigit, img_key))
                if digits:
                    int_val = int(digits)

            candidate_keys = []
            if int_val is not None:
                candidate_keys.extend([
                    f"COCO_train2014_{int_val:012d}.jpg",
                    f"COCO_val2014_{int_val:012d}.jpg",
                    f"{int_val:012d}.jpg",
                    int_val,
                    str(int_val)
                ])

            candidate_keys.append(os.path.basename(str(img_key)))
            if not str(img_key).endswith(".jpg"):
                candidate_keys.append(f"{img_key}.jpg")

            found_feat = None
            for cand in candidate_keys:
                if cand in self.features_dict:
                    found_feat = self.features_dict[cand]
                    break

            if found_feat is not None:
                feat = found_feat
            else:
                raise KeyError(
                    f"Image key '{img_key}' was not found in features.pt dictionary. "
                    f"Sample available keys: {list(self.features_dict.keys())[:3]}"
                )

        # Ensure feature is a FloatTensor of shape [2048]
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        elif feat.dtype != torch.float32:
            feat = feat.to(torch.float32)

        return feat

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        item = self.samples[idx]

        # 1. Retrieve cached image feature [2048]
        image_feature = self._lookup_feature(item["image"])

        # 2. Tokenize and pad/truncate question -> LongTensor [20]
        question_ids = encode_question(item["question"], self.vocab, max_len=self.max_len)

        # 3. Target class index (integer in 0..2999)
        label = item["label"]

        # 4. Return sample index so validation can look up 10 answers and answer_type
        return image_feature, question_ids, label, idx


# ==============================================================================
# DATA LOADING & FILTERING
# ==============================================================================
def format_coco_image_name(image_id: Any, split_prefix: str = "train2014") -> str:
    """
    Loads raw questions JSON file and returns standardized list of sample dicts.
    Formats image_id as a zero-padded 12-digit COCO filename.
    Example: 42 -> COCO_train2014_000000000042.jpg
    """
    if isinstance(image_id, int) or (isinstance(image_id, str) and image_id.isdigit()):
        return f"COCO_{split_prefix}_{int(image_id):012d}.jpg"
    return str(image_id)


def load_raw_questions(questions_path: str, annotations_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads questions and optional annotations files.
    Joins on question_id as specified by official VQA v2.
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
    # Determine default split prefix ('val2014' vs 'train2014') from filename
    split_prefix = "val2014" if "val" in questions_path.lower() else "train2014"

    # Merge annotations if provided
    if annotations_path:
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")
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
        # Resolve image identifier (formatted as COCO zero-padded 12 digits)
        raw_img = item.get("image_id") if "image_id" in item else (item.get("image") or item.get("image_name") or "")
        img_filename = format_coco_image_name(raw_img, split_prefix=split_prefix)

        # Resolve question text
        ques = item.get("question") or item.get("question_str") or item.get("text") or ""

        # Resolve multiple_choice_answer (primary training label)
        mc_ans = ""
        if "multiple_choice_answer" in item and isinstance(item["multiple_choice_answer"], str):
            mc_ans = item["multiple_choice_answer"].strip().lower()
        elif "answer" in item and isinstance(item["answer"], str):
            mc_ans = item["answer"].strip().lower()

        # Resolve list of 10 annotator answers for VQA accuracy calculation
        annotator_answers = []
        if "answers" in item and isinstance(item["answers"], list):
            for a in item["answers"]:
                if isinstance(a, dict) and "answer" in a:
                    annotator_answers.append(a["answer"].strip().lower())
                elif isinstance(a, str):
                    annotator_answers.append(a.strip().lower())
        if not annotator_answers and mc_ans:
            annotator_answers = [mc_ans]

        # Resolve answer_type ('yes/no', 'number', 'other')
        answer_type = item.get("answer_type", "other")
        if not isinstance(answer_type, str) or not answer_type:
            answer_type = "other"
        answer_type = answer_type.strip().lower()

        standardized.append({
            "image": img_filename,
            "image_id": raw_img,
            "question": str(ques).strip(),
            "answer": mc_ans,
            "all_answers": annotator_answers,
            "answer_type": answer_type
        })

    return standardized


def subsample_by_images(samples: List[Dict[str, Any]], num_images: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Subsamples dataset to N unique images, keeping ALL questions for those N images.
    """
    image_groups = defaultdict(list)
    for sample in samples:
        image_groups[sample["image"]].append(sample)

    unique_images = list(image_groups.keys())
    unique_images.sort()
    rng = random.Random(seed)
    rng.shuffle(unique_images)

    selected_images = set(unique_images[:num_images])
    filtered_samples = [s for img in selected_images for s in image_groups[img]]
    return filtered_samples


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
    unique_images.sort()

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
    Matches the official VQA v2 10-answer format and answer_type distribution.
    """
    print("[FAKE MODE] Generating synthetic questions, answers, and image features...")
    print("[FAKE MODE] Generating synthetic VQA v2 format data...")

    # 1. Load or synthesize answers (3000 classes)
    # 1. Load or synthesize answers
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
    image_names = [f"COCO_train2014_{i:012d}.jpg" for i in range(num_images)]
    for img_name in image_names:
        features_dict[img_name] = torch.randn(2048, dtype=torch.float32)

    # 4. Synthesize questions distributed across these images
    # 4. Synthesize questions distributed across images with 10 annotators each
    raw_samples = []
    vocab_words = [w for w in vocab.keys() if w not in ("<pad>", "<unk>")]
    answer_types = ["yes/no", "number", "other"]
    rng = random.Random(42)

    for i in range(num_samples):
        img_name = image_names[i % num_images]
        # Pick 3 to 7 random words from vocab
        q_len = rng.randint(3, 7)
        q_words = [rng.choice(vocab_words) for _ in range(q_len)]
        question_text = " ".join(q_words) + " ?"

        # Pick an answer: 90% in top answers, 10% out-of-vocab to test skipping
        if rng.random() < 0.90:
            mc_ans = rng.choice(answers[:500])
        else:
            mc_ans = f"rare_unseen_answer_{i}"

        # Generate 10 annotator answers (mostly consensus with some variation)
        ten_answers = [mc_ans] * rng.randint(4, 9)
        while len(ten_answers) < 10:
            ten_answers.append(rng.choice(answers[:50]))

        raw_samples.append({
            "image": img_name,
            "image_id": i % num_images,
            "question": question_text,
            "answer": mc_ans,
            "all_answers": ten_answers,
            "answer_type": rng.choice(answer_types)
        })

    return raw_samples, features_dict, vocab, answers


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_model(config: Dict[str, Any]):
    print("=" * 70)
    print("VQA MODEL TRAINING PIPELINE")
    print("VQA MODEL TRAINING PIPELINE (Official VQA v2 Spec)")
    print("=" * 70)

    # Select compute device: CUDA GPU -> Apple MPS -> CPU
    # Select compute device
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
        raw_train_samples, features_dict, vocab, answers = generate_fake_data(
            num_samples=target_samples,
            vocab_path=config["vocab_path"],
            answers_path=config["answers_path"]
        )
        val_features_dict = features_dict
        raw_val_samples = None
    else:
        print(f"Loading vocabulary from: {config['vocab_path']}")
        with open(config["vocab_path"], "r", encoding="utf-8") as f:
            vocab = json.load(f)

        print(f"Loading top answers from: {config['answers_path']}")
        with open(config["answers_path"], "r", encoding="utf-8") as f:
            answers = json.load(f)

        print(f"Loading image features from: {config['features_path']}")
        print(f"Loading train image features from: {config['features_path']}")
        features_dict = torch.load(config["features_path"], map_location="cpu")
        if not isinstance(features_dict, dict):
            raise TypeError("Expected features.pt to contain a dict of {image_name: tensor}.")

        print(f"Loading train questions from: {config['questions_path']}")
        if config["annotations_path"]:
            print(f"Loading train annotations from: {config['annotations_path']}")
        raw_train_samples = load_raw_questions(config["questions_path"], config["annotations_path"])

        # Check for optional separate validation set (e.g. val2014)
        if config["val_questions_path"]:
            print(f"Loading validation questions from: {config['val_questions_path']}")
            raw_val_samples = load_raw_questions(config["val_questions_path"], config["val_annotations_path"])
            if config["val_features_path"]:
                print(f"Loading validation image features from: {config['val_features_path']}")
                val_features_dict = torch.load(config["val_features_path"], map_location="cpu")
            else:
                val_features_dict = features_dict
        else:
            raw_val_samples = None
            val_features_dict = features_dict

    print(f"Loaded {len(raw_train_samples):,} raw training questions.")
    print(f"Vocabulary size: {len(vocab):,} tokens.")
    print(f"Number of answer classes: {len(answers):,} classes.")

    # -------------------------------------------------------------------------
    # 2. FILTER SAMPLES & MAP LABELS
    # -------------------------------------------------------------------------
    ans_to_id = {ans: idx for idx, ans in enumerate(answers)}

    def filter_samples(samples: List[Dict[str, Any]], desc: str = "Train") -> Tuple[List[Dict[str, Any]], int]:
        valid = []
        skipped = 0
        for s in samples:
            ans = s["answer"]
            if ans in ans_to_id:
                valid.append({**s, "label": ans_to_id[ans]})
            else:
                skipped += 1
        print(f"[{desc} Filter] Retained: {len(valid):,} | Skipped: {skipped:,} (not in top {len(answers)} answers).")
        return valid, skipped

    valid_train, _ = filter_samples(raw_train_samples, desc="Train")

    if not valid_train:
        raise RuntimeError("No valid samples remained after filtering! Check answers.json.")
    # Subsample to N training images if requested (--limit_images N)
    if config["limit_images"] is not None and config["limit_images"] > 0:
        valid_train = subsample_by_images(valid_train, config["limit_images"], seed=seed)
        unique_imgs = len({s["image"] for s in valid_train})
        print(f"Applied --limit_images: Subsampled to {unique_imgs} images ({len(valid_train):,} questions).")

    # Apply --limit N if requested
    # Apply sample limit if requested (--limit N)
    if config["limit"] is not None and config["limit"] > 0:
        valid_train = valid_train[:config["limit"]]
        print(f"Applied --limit: Training on the first {len(valid_train)} samples.")

    # -------------------------------------------------------------------------
    # 3. SPLIT TRAIN / VAL BY IMAGE
    # 3. ASSEMBLE TRAIN / VAL SPLITS
    # -------------------------------------------------------------------------
    if raw_val_samples is not None:
        # User provided an explicit validation set (e.g. val2014)
        valid_val, _ = filter_samples(raw_val_samples, desc="Val")
        train_samples = valid_train
        val_samples = valid_val
        n_train_imgs = len({s["image"] for s in train_samples})
        n_val_imgs = len({s["image"] for s in val_samples})
        print("-" * 70)
        print("Explicit Dataset Splits (Train & Separate Val):")
        print(f"  Train: {len(train_samples):,} samples across {n_train_imgs:,} unique images")
        print(f"  Val:   {len(val_samples):,} samples across {n_val_imgs:,} unique images")
        print("-" * 70)
    else:
        # Split train by IMAGE (held-out slice of train, guaranteed disjoint)
        train_samples, val_samples, n_train_imgs, n_val_imgs = split_by_image(
            valid_train,
            val_split=config["val_split"],
            seed=seed
        )
        print("-" * 70)
        print("Disjoint Image-Level Split (Held-Out Slice of Train):")
        print(f"  Train: {len(train_samples):,} samples across {n_train_imgs:,} unique images")
        print(f"  Val:   {len(val_samples):,} samples across {n_val_imgs:,} unique images")
        print("-" * 70)

    print("-" * 70)
    print("Train/Val Split Summary (Strict Image-Level Disjoint Split):")
    print(f"  Train: {len(train_samples):,} samples across {n_train_imgs:,} unique images")
    print(f"  Val:   {len(val_samples):,} samples across {n_val_imgs:,} unique images")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # 4. CREATE DATA LOADERS
    # -------------------------------------------------------------------------
    train_dataset = VQADataset(train_samples, features_dict, vocab, max_len=config["max_len"])
    val_dataset = VQADataset(val_samples, val_features_dict, vocab, max_len=config["max_len"])

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
    best_val_vqa_acc = -1.0
    patience = config["early_stopping_patience"]
    patience_counter = 0

    print("Starting training loop (up to {} epochs)...".format(config["num_epochs"]))
    print("=" * 70)

    for epoch in range(1, config["num_epochs"] + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for img_feats, q_tokens, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
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
        # --- VALIDATION PHASE (VQA Accuracy: min(matches/3, 1)) ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        correct_predictions = 0

        # Metrics tracking
        all_vqa_scores = []
        scores_by_type = defaultdict(list)
        correct_exact_match = 0
        total_val_samples = 0

        with torch.no_grad():
            for img_feats, q_tokens, labels, sample_indices in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  "):
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
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                batch_indices = sample_indices.tolist()

                for pred_class_idx, sample_idx in zip(preds, batch_indices):
                    pred_answer_str = answers[pred_class_idx]
                    sample_data = val_samples[sample_idx]

                    # Official VQA Accuracy: min(matches / 3.0, 1.0)
                    sample_score = compute_vqa_accuracy(pred_answer_str, sample_data["all_answers"])
                    all_vqa_scores.append(sample_score)

                    # Group by answer_type ('yes/no', 'number', 'other')
                    atype = sample_data.get("answer_type", "other")
                    scores_by_type[atype].append(sample_score)

                    # Also track exact match with multiple_choice_answer
                    if pred_answer_str == sample_data["answer"]:
                        correct_exact_match += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_accuracy = (correct_predictions / max(1, total_val_samples)) * 100.0
        overall_vqa_acc = (sum(all_vqa_scores) / max(1, len(all_vqa_scores))) * 100.0
        exact_match_acc = (correct_exact_match / max(1, total_val_samples)) * 100.0

        # Calculate per-answer_type accuracy
        type_accs = {}
        for atype in ["yes/no", "number", "other"]:
            scores = scores_by_type.get(atype, [])
            type_accs[atype] = (sum(scores) / max(1, len(scores))) * 100.0 if scores else 0.0

        # Print epoch metrics
        print(
            f"Epoch {epoch:02d}/{config['num_epochs']:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% ({correct_predictions}/{total_val_samples})"
            f"Val VQA Acc: {overall_vqa_acc:.2f}% | "
            f"Exact Match: {exact_match_acc:.2f}%"
        )
        print(
            f"  --> Per Answer Type: "
            f"Yes/No: {type_accs['yes/no']:.2f}% ({len(scores_by_type['yes/no'])}) | "
            f"Number: {type_accs['number']:.2f}% ({len(scores_by_type['number'])}) | "
            f"Other: {type_accs['other']:.2f}% ({len(scores_by_type['other'])})"
        )

        # Save checkpoint every epoch
        # Save checkpoint each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_accuracy,
            "val_vqa_acc": overall_vqa_acc,
            "val_exact_match": exact_match_acc,
            "per_type_acc": type_accs,
            "vocab_size": len(vocab),
            "num_classes": len(answers),
        }
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_ckpt_path)

        # Keep the best checkpoint by validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        # Track best model by official VQA accuracy
        if overall_vqa_acc > best_val_vqa_acc:
            best_val_vqa_acc = overall_vqa_acc
            patience_counter = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_ckpt_path)
            print(f"  --> Saved new best checkpoint: {best_ckpt_path} (Val Acc: {best_val_acc:.2f}%)")
            print(f"  --> Saved new best checkpoint: {best_ckpt_path} (VQA Acc: {best_val_vqa_acc:.2f}%)")
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
    print(f"Training completed. Best Validation VQA Accuracy: {best_val_vqa_acc:.2f}%")
    print(f"Best model saved at: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train VQA Language and Classifier Pipeline.")

    # Paths (configurable via CLI or top config block)
    # Train data paths
    parser.add_argument("--questions", type=str, default=DEFAULT_CONFIG["questions_path"],
                        help="Path to training questions JSON file.")
    parser.add_argument("--annotations", type=str, default=DEFAULT_CONFIG["annotations_path"],
                        help="Optional path to training annotations JSON file.")
    parser.add_argument("--features", type=str, default=DEFAULT_CONFIG["features_path"],
                        help="Path to features.pt file containing image feature dict.")

    # Validation data paths (optional: if omitted, splits train set by image)
    parser.add_argument("--val_questions", type=str, default=DEFAULT_CONFIG["val_questions_path"],
                        help="Optional path to validation questions JSON (e.g. val2014).")
    parser.add_argument("--val_annotations", type=str, default=DEFAULT_CONFIG["val_annotations_path"],
                        help="Optional path to validation annotations JSON.")
    parser.add_argument("--val_features", type=str, default=DEFAULT_CONFIG["val_features_path"],
                        help="Optional path to validation features.pt.")

    # Shared vocabulary and answers
    parser.add_argument("--vocab", type=str, default=DEFAULT_CONFIG["vocab_path"],
                        help="Path to vocab.json.")
    parser.add_argument("--answers", type=str, default=DEFAULT_CONFIG["answers_path"],
                        help="Path to answers.json.")
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
    # Subsampling & Debug flags
    parser.add_argument("--limit_images", type=int, default=DEFAULT_CONFIG["limit_images"],
                        help="Subsample to N training images (keeps all their questions).")
    parser.add_argument("--limit", type=int, default=DEFAULT_CONFIG["limit"],
                        help="Limit training to only N total samples.")
    parser.add_argument("--fake", action="store_true",
                        help="Use synthetic questions and random features for dry-run testing.")

    args = parser.parse_args()

    config = {
        "questions_path": args.questions,
        "annotations_path": args.annotations,
        "features_path": args.features,
        "val_questions_path": args.val_questions,
        "val_annotations_path": args.val_annotations,
        "val_features_path": args.val_features,
        "vocab_path": args.vocab,
        "answers_path": args.answers,
        "checkpoint_dir": args.checkpoint_dir,
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "dropout": args.dropout,
        "max_len": DEFAULT_CONFIG["max_len"],
        "seed": DEFAULT_CONFIG["seed"],
        "limit_images": args.limit_images,
        "limit": args.limit,
        "fake": args.fake,
        "device": args.device,
    }

    train_model(config)


if __name__ == "__main__":
    main()
