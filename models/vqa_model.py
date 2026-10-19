"""
VQA Model Architecture.

Combines visual features (ResNet-50 pooled features) and textual features (LSTM output)
using multimodal fusion (element-wise multiplication) to classify into the top 3000 answers.

Architecture breakdown:
1. Image Projection: Linear(1024, 1024) -> Tanh -> [B, 1024]
2. Question Encoder: Embedding(vocab_size, 300) -> LSTM(300, 1024) -> Tanh -> [B, 1024]
3. Multimodal Fusion: Element-wise multiplication (img_proj * ques_emb) -> [B, 1024]
4. Classifier: Linear(1024, 1024) -> ReLU -> Dropout(0.5) -> Linear(1024, 3000) -> [B, 3000]

Contract:
- forward(image_features, question_ids)
  image_features: Tensor [B, 1024]
  question_ids: LongTensor [B, 20]
  returns: Tensor [B, 3000]
"""

from typing import Optional
import torch
import torch.nn as nn

from text_encoder.question_encoder import QuestionEncoder


class VQAModel(nn.Module):
    """
    Complete VQA Model performing image-question feature extraction,
    element-wise fusion, and multi-class answer classification.
    """

    def __init__(
        self,
        vocab_size: int,
        image_dim: int = 1024,
        text_dim: int = 1024,
        num_classes: int = 3000,
        dropout_prob: float = 0.5,
        question_encoder: Optional[QuestionEncoder] = None
    ):
        """
        Args:
            vocab_size (int): Size of question vocabulary (needed if question_encoder not given).
            image_dim (int): Dimensionality of input image features (default: 1024).
            text_dim (int): Common multimodal projection dimension (default: 1024).
            num_classes (int): Number of answer classes (default: 3000).
            dropout_prob (float): Dropout probability for regularization (default: 0.5).
            question_encoder (Optional[QuestionEncoder]): Pre-instantiated QuestionEncoder.
        """
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes

        # 1. Question Encoder: Embedding(vocab_size, 300) -> LSTM(300, 1024) -> tanh
        if question_encoder is not None:
            self.question_encoder = question_encoder
        else:
            self.question_encoder = QuestionEncoder(
                vocab_size=vocab_size,
                embed_dim=300,
                hidden_dim=text_dim,
                padding_idx=0
            )

        # 2. Image projection: Linear(1024, 1024)
        # Projects pooled image features into the common 1024-dimensional space
        self.image_linear = nn.Linear(image_dim, text_dim)

        # 3. Classifier MLP:
        # Linear(1024, 1024) -> ReLU -> Dropout(0.5) -> Linear(1024, 3000)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(text_dim, num_classes)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        question_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for VQA prediction.

        Args:
            image_features (torch.Tensor): Pre-extracted image features [B, 1024].
            question_ids (torch.LongTensor): Token IDs [B, 20].

        Returns:
            torch.Tensor: Unnormalized prediction logits of shape [B, 3000].
        """
        # Step 1: Project image features to 1024-dim and apply tanh activation
        # [B, 1024] -> [B, 1024]
        img_proj = torch.tanh(self.image_linear(image_features))

        # Step 2: Encode question tokens to 1024-dim
        # [B, 20] -> [B, 1024]
        ques_emb = self.question_encoder(question_ids)

        # Step 3: Multimodal fusion via element-wise multiplication (Hadamard product)
        # Both tensors have shape [B, 1024], resulting in [B, 1024]
        fused = img_proj * ques_emb

        # Step 4: Pass fused representation through the classification head
        # [B, 1024] -> [B, 3000]
        logits = self.classifier(fused)

        return logits

