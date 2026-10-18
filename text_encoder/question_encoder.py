"""
Question Encoder module for Visual Question Answering (VQA).

Architecture:
- nn.Embedding(vocab_size, 300, padding_idx=0)
- nn.LSTM(input_size=300, hidden_size=1024, batch_first=True)
- Tanh activation on the final hidden state.

Contract:
- Input: question_ids of shape [B, seq_len] (typically [B, 20]), dtype=torch.long
- Output: question representation tensor of shape [B, 1024]
"""

import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    """
    Encodes question token IDs into a 1024-dimensional feature vector.

    Attributes:
        embedding (nn.Embedding): Maps discrete word IDs to dense 300-d vectors.
        lstm (nn.LSTM): Recurrent network processing word embeddings across time steps.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size (int): Total number of unique tokens in the vocabulary.
            embed_dim (int): Word embedding dimensionality (default: 300).
            hidden_dim (int): LSTM hidden state dimensionality (default: 1024).
            padding_idx (int): ID of <pad> token to ignore in embedding updates (default: 0).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 1. Word embedding layer: maps token IDs -> 300-dim continuous vectors
        # padding_idx=0 ensures <pad> embeddings are initialized to 0 and not updated.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # 2. Unidirectional single-layer LSTM: processes sequence of embeddings
        # batch_first=True means inputs/outputs have shape [Batch, Seq_len, Features]
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, question_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QuestionEncoder.

        Args:
            question_ids (torch.LongTensor): Token IDs of shape [B, seq_len].

        Returns:
            torch.Tensor: Question embedding tensor of shape [B, 1024].
        """
        # Step 1: Lookup embeddings for each word in the sequence
        # Input shape: [B, seq_len] -> Output shape: [B, seq_len, 300]
        embedded = self.embedding(question_ids)

        # Step 2: Pass embedded sequence through the LSTM
        # output shape: [B, seq_len, 1024]
        # h_n shape: [num_layers (1), B, 1024]
        # c_n shape: [num_layers (1), B, 1024]
        _, (h_n, _) = self.lstm(embedded)

        # Step 3: Extract the final hidden state of the LSTM
        # h_n[-1] has shape [B, 1024]
        final_hidden = h_n[-1]

        # Step 4: Apply tanh non-linearity as required by the contract
        # Returns shape [B, 1024]
        out = torch.tanh(final_hidden)
        return out

