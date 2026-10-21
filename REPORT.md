# Visual Question Answering (VQA) — System Architecture & Methodology Report

**Module**: Language Modeling, Multimodal Fusion, and Classification Pipeline  
**Role**: Person B (Language & Classifier Half)  
**Teammate**: Person A (Visual Feature Extraction, Image Dataset, Evaluation & Web Demo)  

---

## 1. Executive Summary

Visual Question Answering (VQA) requires a machine learning system to comprehend an open-ended natural language question about an input image and produce an accurate, concise textual answer.

This report documents the architectural design, mathematical formulations, training methodology, and design trade-offs implemented for **Person B's** subsystem. The subsystem consumes pre-extracted 2048-dimensional pooled visual representations (extracted via ResNet-50 by Person A) and raw textual question strings, transforming them into a 3000-class categorical prediction over the most frequent dataset answers.

---

## 2. Team Split & Contractual Interface

To ensure seamless integration between Person A and Person B without dependency collisions, the project adheres to strict interfacial contracts:

```
+-----------------------------------------------------------------------------------------+
|                               PERSON A (Vision Half)                                    |
|  - Raw Images -> ResNet-50 Backbone -> Pooled Feature Tensor [B, 2048]                  |
|  - Serialized Feature Storage: features.pt ({image_id: Tensor[2048]})                   |
|  - Evaluation Script (evaluate.py), PyTorch Dataset integration, and Demo UI (app/)     |
+-----------------------------------------------------------------------------------------+
                                           │
                                           │ features.pt [B, 2048]
                                           ▼
+-----------------------------------------------------------------------------------------+
|                               PERSON B (Language & Model Half)                          |
|  - Text Preprocessing & Vocabulary Generation (data/build_vocab.py -> vocab.json)       |
|  - Top 3000 Candidate Answer Extraction (data/build_vocab.py -> answers.json)           |
|  - Sequence Tokenization (text_encoder/tokenizer.py -> LongTensor [B, 20])              |
|  - Recurrent Question Encoder (text_encoder/question_encoder.py -> Tensor [B, 1024])   |
|  - Multimodal Fusion & Classification Head (models/vqa_model.py -> Logits [B, 3000])    |
|  - End-to-End Training & Image-Disjoint Validation Pipeline (train.py)                  |
|  - Architectural Unit Testing & Contract Verification (tests/test_shapes.py)            |
+-----------------------------------------------------------------------------------------+
```

### Shared Interface Contracts
1. **Question Tokens**: `LongTensor [B, 20]` (lowercased, whitespace split, `<pad>=0`, `<unk>=1`, truncated/padded to length 20).
2. **Question Representation**: `Tensor [B, 1024]` ($\tanh$ activation on final LSTM hidden state).
3. **Image Feature Input**: `Tensor [B, 2048]` (pre-extracted ResNet-50 pooled features).
4. **Answer Classification**: `Tensor [B, 3000]` unnormalized logits matching the top 3,000 answer classes in `answers.json`.
5. **Shared Artifacts**:
   - `vocab.json`: Mapping of words with $\ge 2$ occurrences to unique integer IDs.
   - `answers.json`: Ordered JSON list of 3,000 answer strings (`index = class ID`).

---

## 3. Methodology & System Architecture

### 3.1 Text Tokenization & Vocabulary Construction
- **Cleaning & Tokenization**: Questions are lowercased and split purely on whitespace as specified by the contract. This preserves tokenization consistency between Person A's evaluation scripts and Person B's training pipeline.
- **Vocabulary Filtering**: Rare words occurring only once ($\text{count} < 2$) across the training questions are filtered out to avoid overfitting to typos and noise. Filtered tokens map to the special out-of-vocabulary token `<unk>` (ID 1). Zero-padding maps to `<pad>` (ID 0).
- **Sequence Normalization**: Sequences shorter than 20 words are right-padded with `<pad>`. Sequences longer than 20 words are truncated to the first 20 tokens.

### 3.2 Question Encoder Architecture
The Question Encoder transforms a sequence of discrete token IDs $[w_1, w_2, \dots, w_T]$ ($T=20$) into a dense 1024-dimensional semantic embedding:
1. **Embedding Layer**: Discrete tokens are mapped to continuous 300-dimensional vectors via an embedding matrix $E \in \mathbb{R}^{|V| \times 300}$. The padding token index (`padding_idx=0`) is explicitly frozen at zero gradient updates.
2. **LSTM Recurrent Network**: The 300-d embeddings are fed into a single-layer, unidirectional Long Short-Term Memory (LSTM) network with hidden size $H = 1024$:
   $$h_t, c_t = \text{LSTM}(e_t, (h_{t-1}, c_{t-1}))$$
3. **Activation**: The final hidden state $h_T$ summarizes the entire syntactic and semantic question context. A hyperbolic tangent non-linearity is applied:
   $$q_{\text{emb}} = \tanh(h_T) \in [-1, 1]^{1024}$$

### 3.3 Visual Feature Projection
The input visual feature $v \in \mathbb{R}^{2048}$ represents the pooled feature map from the penultimate layer of ResNet-50. To map visual features into the same metric space as the question embeddings:
$$v_{\text{proj}} = \tanh(W_v v + b_v) \in [-1, 1]^{1024}$$
where $W_v \in \mathbb{R}^{1024 \times 2048}$ and $b_v \in \mathbb{R}^{1024}$.

> **Viva Note**: Why apply $\tanh$ to both visual and textual projections?  
> Applying $\tanh$ scales both modalities into the identical dynamic range $[-1, 1]$. This prevents visual feature magnitudes from dominating question feature magnitudes during multimodal fusion.

### 3.4 Multimodal Fusion: Element-Wise Multiplication
Multimodal fusion combines visual and textual cues into a unified representation. The fused representation $z$ is computed using the Hadamard product (element-wise multiplication):
$$z = v_{\text{proj}} \odot q_{\text{emb}} \in \mathbb{R}^{1024}$$
where $z_i = (v_{\text{proj}})_i \cdot (q_{\text{emb}})_i$.

**Theoretical Justification**:
- **Feature-Wise Gating**: The question representation acts as an attention filter over the visual dimensions. If the question asks *"What color is the car?"*, visual channels sensitive to color and vehicles are preserved/amplified, while irrelevant spatial/background features are attenuated.
- **Parameter Efficiency**: Unlike concatenative fusion ($[v; q] \in \mathbb{R}^{2048}$), element-wise multiplication preserves a compact 1024-d dimension without introducing additional weight matrices, mitigating overfitting on small-to-medium datasets.

### 3.5 Classification Head
The fused vector $z$ is passed to a multi-layer perceptron (MLP) classification head:
1. **Linear Layer**: $\mathbb{R}^{1024} \rightarrow \mathbb{R}^{1024}$
2. **Non-Linearity**: $\text{ReLU}(x) = \max(0, x)$
3. **Regularization**: $\text{Dropout}(p=0.5)$ to prevent co-adaptation of features during training.
4. **Output Linear Projection**: $\mathbb{R}^{1024} \rightarrow \mathbb{R}^{3000}$, yielding unnormalized logits $\hat{y} \in \mathbb{R}^{3000}$.

---

## 4. Dimensional Contracts & Tensor Shapes Table

| Stage | Tensor Variable | Data Type | Dimensions | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Input Image** | `image_features` | `torch.float32` | $[B, 2048]$ | Pre-extracted ResNet-50 pooled features |
| **Input Question** | `question_ids` | `torch.long` | $[B, 20]$ | Tokenized word IDs (padded with 0, unk=1) |
| **Word Embedding** | `embedded` | `torch.float32` | $[B, 20, 300]$ | Dense word representation sequence |
| **Question Encoding**| `ques_emb` | `torch.float32` | $[B, 1024]$ | $\tanh(\text{final LSTM hidden state})$ |
| **Image Projection** | `img_proj` | `torch.float32` | $[B, 1024]$ | $\tanh(\text{Linear}(2048, 1024))$ |
| **Multimodal Fusion** | `fused` | `torch.float32` | $[B, 1024]$ | Element-wise product: `img_proj * ques_emb` |
| **Classifier Hidden**| `hidden` | `torch.float32` | $[B, 1024]$ | $\text{Dropout}_{0.5}(\text{ReLU}(\text{Linear}(1024, 1024)))$ |
| **Model Output** | `logits` | `torch.float32` | $[B, 3000]$ | Final unnormalized classification scores |

---

## 5. Training & Validation Methodology

### 5.1 Strict Image-Disjoint Split
A common flaw in naive VQA training is random splitting across questions. Because multiple questions often query the same image (e.g., *"What is on the table?"* and *"What color is the cup?"*), randomly splitting questions would leak identical image features into both the training and validation sets.

**Implementation**:
1. Group all validated samples by unique image identifier.
2. Partition the set of unique image IDs into 80% train and 20% validation.
3. Construct splits such that:
   $$\text{Images}_{\text{train}} \cap \text{Images}_{\text{val}} = \emptyset$$
This guarantees that validation accuracy reflects genuine generalization to unseen visual scenes.

### 5.2 Out-of-Vocabulary Answer Filtering
Because VQA is cast as a 3000-way classification problem, samples whose answers fall outside the top 3,000 frequent answers cannot be supervised by the classifier. The pipeline automatically filters and tallies these samples, reporting exact retention numbers in logs.

### 5.3 Loss Function & Optimization
- **Loss**: Categorical Cross-Entropy Loss over 3,000 classes:
  $$\mathcal{L} = -\sum_{c=1}^{3000} y_c \log \left(\frac{e^{\hat{y}_c}}{\sum_{j=1}^{3000} e^{\hat{y}_j}}\right)$$
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=10^{-3}$).
- **Batch Size**: 64.
- **Early Stopping**: Monitored on validation accuracy with `patience=3` epochs to avoid overfitting.
- **Early Stopping**: Monitored on validation VQA accuracy with `patience=3` epochs to avoid overfitting.
- **Checkpointing**: Every epoch is checkpointed, with `best_model.pt` automatically preserved.

### 5.4 Official VQA Evaluation Metric (10 Annotators)
For evaluation, naive exact match against a single ground truth is replaced by the official VQA competition accuracy metric across all 10 human annotators:
$$\text{Accuracy}_{\text{VQA}} = \min\left(\frac{\sum_{i=1}^{10} \mathbb{I}(\hat{a} = a_i)}{3}, 1.0\right)$$
where $\hat{a}$ is the predicted answer string and $\{a_1, \dots, a_{10}\}$ are the 10 human annotations.

**Per-Answer-Type Breakdown**:
Accuracy is tracked and reported across standard question categories:
- **Yes/No**: Binary affirmation/negation questions.
- **Number**: Quantitative counting questions.
- **Other**: Open-ended attribute, object, and color queries.

---

## 6. Software Dependencies (`requirements.txt`)

| Package | Version Requirement | Purpose | Rationale |
| :--- | :--- | :--- | :--- |
| `torch` | $\ge 2.0.0$ | Core deep learning framework | GPU/MPS acceleration, autograd, LSTM, tensor ops |
| `numpy` | $\ge 1.22.0$ | Numerical calculations | Efficient array operations and metric calculations |
| `tqdm` | $\ge 4.65.0$ | Epoch progress indicators | Real-time train/val loss monitoring in terminal |

---

## 7. Execution Guide & Google Colab Workflow

### 7.1 Dry-Run Verification (Synthetic Mode)
Before Person A finishes feature extraction, test the end-to-end pipeline using synthetic data:
```bash
# 1. Test tensor shapes & dimensions
python3 -m unittest tests/test_shapes.py

# 2. Test vocabulary builder
python3 data/build_vocab.py --fake

# 3. Test training loop with 200 synthetic samples
python3 train.py --fake --limit 200
```

### 7.2 Real Training on Google Colab
On Google Colab, mount Google Drive so large feature files and model checkpoints persist outside the ephemeral virtual machine:
### 7.2 Real Training on Google Colab with Official VQA v2 Data
Google Drive path: `MyDrive/VQA Project/dataset/`

```python
from google.colab import drive
drive.mount('/content/drive')

# Step 1: Generate vocabulary and answer set
!python data/build_vocab.py --questions /content/drive/MyDrive/VQA/questions.json \
                           --vocab_out /content/drive/MyDrive/VQA/vocab.json \
                           --answers_out /content/drive/MyDrive/VQA/answers.json
DRIVE_DIR = "/content/drive/MyDrive/VQA Project"

# Step 2: Train VQA Model
!python train.py --questions /content/drive/MyDrive/VQA/questions.json \
                --features /content/drive/MyDrive/VQA/features.pt \
                --vocab /content/drive/MyDrive/VQA/vocab.json \
                --answers /content/drive/MyDrive/VQA/answers.json \
                --checkpoint_dir /content/drive/MyDrive/VQA/checkpoints \
                --batch_size 64 \
                --epochs 15 \
                --lr 0.001
# Step 1: Generate vocabulary & answers from official train files
!python data/build_vocab.py \
    --questions "$DRIVE_DIR/dataset/v2_OpenEnded_mscoco_train2014_questions.json" \
    --annotations "$DRIVE_DIR/dataset/v2_mscoco_train2014_annotations.json" \
    --vocab_out "$DRIVE_DIR/vocab.json" \
    --answers_out "$DRIVE_DIR/answers.json"
#Output of build_vocab.py
Dataset size (questions):            443,757
Number of distinct answers:          22,531
Questions answered by top 3000:     413,036 / 443,757 (93.08%)

# Step 2: Train with disjoint image split or separate val2014
!python train.py \
    --questions "$DRIVE_DIR/dataset/v2_OpenEnded_mscoco_train2014_questions.json" \
    --annotations "$DRIVE_DIR/dataset/v2_mscoco_train2014_annotations.json" \
    --features "$DRIVE_DIR/features.pt" \
    --val_questions "$DRIVE_DIR/dataset/v2_OpenEnded_mscoco_val2014_questions.json" \
    --val_annotations "$DRIVE_DIR/dataset/v2_mscoco_val2014_annotations.json" \
    --vocab "$DRIVE_DIR/vocab.json" \
    --answers "$DRIVE_DIR/answers.json" \
    --checkpoint_dir "$DRIVE_DIR/checkpoints" \
    --limit_images 5000 \
    --batch_size 64 \
    --epochs 15 \
    --lr 0.001
```

---

## 8. Viva Voce Defense Guide (Anticipated Questions & Answers)

1. **Q: Why treat VQA as classification rather than sequence generation (like GPT or an autoregressive decoder)?**  
   *A*: Most VQA questions (e.g., in VQA v2) require concise 1-3 word answers (colors, counts, yes/no). Casting the top 3,000 answers as classification classes covers $>85\%$ of all answers, converges significantly faster, requires far fewer parameters, and avoids open-ended decoding hallucinations.

2. **Q: Why use LSTM instead of a Transformer (like BERT) for the question encoder?**  
   *A*: Questions in VQA are brief (average length 6–8 words, capped at 20). A lightweight LSTM with 300-d embeddings and 1024-d hidden size captures sequential context with minimal computational overhead and trains stably from scratch without requiring large-scale pre-training.

3. **Q: How does the model handle words unseen during training?**  
   *A*: During vocabulary generation, words with count $< 2$ are excluded from `vocab.json`. Any unseen word at inference time maps to the reserved `<unk>` token (ID 1), allowing the LSTM to maintain syntactic sequence flow without crashing.

4. **Q: Why is early stopping based on validation accuracy rather than validation loss?**  
   *A*: In highly multi-class problems with a long-tailed answer distribution, cross-entropy loss can fluctuate due to uncalibrated overconfident probability tails, whereas classification accuracy directly measures top-1 exact match, which is the primary evaluation metric.

