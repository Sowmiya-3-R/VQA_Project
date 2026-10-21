# VQA - Language & Multimodal Classification (Person B)

This repository contains the language encoding, multimodal fusion, classification model, vocabulary generation, and training pipeline for Visual Question Answering (VQA).

Image feature extraction and dataset visual pipelines are managed by **Person A**, while this module (**Person B**) handles:
- Question tokenization and sequence encoding via LSTM
- Vocabulary and answer candidate extraction
- Multimodal fusion (element-wise multiplication)
- Answer classification into the top 3,000 frequent answers
- Training pipeline with image-disjoint train/validation split and early stopping

---

## 1. Architectural Contract & Tensor Shapes

All components strictly honor the dimensional contract agreed upon with Person A:
- **Question tokens**: `LongTensor [B, 20]` (lowercased, whitespace split, `<pad>=0`, `<unk>=1`, truncated or padded to length 20).
- **QuestionEncoder output**: `Tensor [B, 1024]` (`tanh` of final LSTM hidden state).
- **Image features**: `Tensor [B, 2048]` (pre-extracted pooled ResNet-50 features from `features.pt`).
- **Image projection**: `Linear(2048, 1024)` + `tanh` -> `Tensor [B, 1024]`.
- **Multimodal fusion**: Element-wise multiplication `[B, 1024] * [B, 1024] -> [B, 1024]`.
- **Classifier**: `Linear(1024, 1024) -> ReLU -> Dropout(0.5) -> Linear(1024, 3000)`.
- **Model output**: `Tensor [B, 3000]` unnormalized logits over top 3,000 answer classes.
- **Shared contract files**:
  - `answers.json`: JSON array of the top 3,000 answers (`index = class ID`).
  - `vocab.json`: JSON dictionary mapping words with $\ge 2$ occurrences to integer IDs (`"<pad>": 0`, `"<unk>": 1`).

---

## 2. Installation

Install required dependencies from the repo root:
```bash
python -m pip install -r requirements.txt
```

If you are already inside a virtual environment, use that environment instead of the system Python.

To run the Gradio app from this repo, use the root file:
```bash
python app.py --checkpoint checkpoints/best_model.pt \
             --vocab vocab.json \
             --answers answers.json \
             --projection projection.pt
```

The repo currently includes the model checkpoint under `checkpoints/`, but it does not include the real `vocab.json`, `answers.json`, and `projection.pt` files needed for a live inference run. You must generate or obtain those files before starting the app.

---

## 3. How to Run Each Script

### A. Build Vocabulary and Answer List (`data/build_vocab.py`)

Scans question data, creates `vocab.json` (frequency $\ge 2$) and `answers.json` (top 3,000 answers), and reports coverage statistics:

```bash
python data/build_vocab.py --questions data/questions.json \
                           --vocab_out vocab.json \
                           --answers_out answers.json \
                           --min_freq 2 \
                           --num_answers 3000
```

*(Optional: If working with official VQA v2 separate annotations, add `--annotations path/to/annotations.json`)*.

### B. Run Contract & Shape Unit Tests (`tests/test_shapes.py`)

Verifies that the `Tokenizer`, `QuestionEncoder`, and `VQAModel` strictly conform to the expected tensor shapes:

```bash
python -m unittest tests/test_shapes.py
```
Or directly:
```bash
python tests/test_shapes.py
```

### C. Training the Model (`train.py`)

#### Dry-run with Synthetic Data (`--fake` mode)
To test the entire training and validation loop before Person A finishes extracting `features.pt`:
```bash
python train.py --fake --limit 200
```
- Generates synthetic 2048-d image features and question-answer pairs in-memory.
- Performs disjoint image splitting, forward/backward passes, metric calculation, early stopping, and checkpoint saving.

#### Full Training on Real Features
#### Full Training on Official VQA v2 Data
```bash
python train.py --questions data/questions.json \
python train.py --questions dataset/v2_OpenEnded_mscoco_train2014_questions.json \
                --annotations dataset/v2_mscoco_train2014_annotations.json \
                --features features.pt \
                --vocab vocab.json \
                --answers answers.json \
                --val_questions dataset/v2_OpenEnded_mscoco_val2014_questions.json \
                --val_annotations dataset/v2_mscoco_val2014_annotations.json \
                --limit_images 5000 \
                --checkpoint_dir checkpoints \
                --batch_size 64 \
                --lr 1e-3 \
                --epochs 15 \
                --patience 3
```

#### Quick Test on Real Data (`--limit N`)
To train on a small subset (e.g. 100 samples) to verify end-to-end data loading:
#### Quick Test on Real Data
To train on only $N$ training images (e.g. 100 images, keeping all their questions):
```bash
python train.py --questions data/questions.json \
python train.py --questions dataset/v2_OpenEnded_mscoco_train2014_questions.json \
                --annotations dataset/v2_mscoco_train2014_annotations.json \
                --features features.pt \
                --limit 100
                --limit_images 100
```

---

## 4. Google Colab / Google Drive Checkpoint Storage

Checkpoints and large data files should never be committed inside the Git repository. On Google Colab, mount Google Drive and set the paths accordingly:
Checkpoints and large data files should never be committed inside the Git repository. On Google Colab, mount Google Drive and run with paths pointing to `VQA Project/`:

```python
from google.colab import drive
drive.mount('/content/drive')

# Run training pointing checkpoints to Drive:
!python train.py --questions /content/drive/MyDrive/VQA/questions.json \
                 --features /content/drive/MyDrive/VQA/features.pt \
                 --vocab /content/drive/MyDrive/VQA/vocab.json \
                 --answers /content/drive/MyDrive/VQA/answers.json \
                 --checkpoint_dir /content/drive/MyDrive/VQA/checkpoints
DRIVE_DIR = "/content/drive/MyDrive/VQA Project"

# Run training pointing checkpoints and dataset to Drive:
!python train.py \
    --questions "$DRIVE_DIR/dataset/v2_OpenEnded_mscoco_train2014_questions.json" \
    --annotations "$DRIVE_DIR/dataset/v2_mscoco_train2014_annotations.json" \
    --features "$DRIVE_DIR/features.pt" \
    --vocab "$DRIVE_DIR/vocab.json" \
    --answers "$DRIVE_DIR/answers.json" \
    --checkpoint_dir "$DRIVE_DIR/checkpoints" \
    --limit_images 5000 \
    --batch_size 64 \
    --epochs 15
```

---

## 5. Architecture & Methodology Report

For the complete technical report detailing the mathematical formulation, design decisions, layer dimensions, package choices, and viva voce question preparation, see [REPORT.md](REPORT.md).


