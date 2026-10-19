"""
Unit tests for tensor shapes and contracts across VQA components.

Contract verification:
1. QuestionEncoder:
   Input [B, 20] LongTensor -> Output [B, 1024] FloatTensor
2. VQAModel:
   Image features [B, 1024] FloatTensor + Question tokens [B, 20] LongTensor -> Output [B, 3000] FloatTensor
3. Tokenizer:
   Input question string -> Output [20] LongTensor with padding and unknown tokens handled.
"""

import os
import sys
import unittest
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_encoder.tokenizer import encode_question, PAD_ID, UNK_ID
from text_encoder.question_encoder import QuestionEncoder
from models.vqa_model import VQAModel


class TestVQAShapes(unittest.TestCase):
    """
    Test suite verifying dimensional contracts for Person B's components.
    """

    def setUp(self):
        self.batch_size = 8
        self.seq_len = 20
        self.vocab_size = 500
        self.image_dim = 1024
        self.hidden_dim = 1024
        self.num_classes = 3000

        # Dummy vocab
        self.vocab = {
            "<pad>": PAD_ID,
            "<unk>": UNK_ID,
            "what": 2,
            "color": 3,
            "is": 4,
            "the": 5,
            "dog": 6,
        }

    def test_tokenizer_shape_and_padding(self):
        """
        Tests encode_question output shape, padding, truncation, and unk mapping.
        """
        # Test 1: Short question (should be padded to 20 with 0)
        q_short = "what color is"
        encoded_short = encode_question(q_short, self.vocab, max_len=self.seq_len)

        self.assertIsInstance(encoded_short, torch.Tensor)
        self.assertEqual(encoded_short.dtype, torch.long)
        self.assertEqual(list(encoded_short.shape), [self.seq_len])
        # First 3 tokens: [2, 3, 4], followed by 17 zeros
        self.assertEqual(encoded_short[:3].tolist(), [2, 3, 4])
        self.assertEqual(encoded_short[3:].tolist(), [PAD_ID] * 17)

        # Test 2: Unknown words mapped to UNK_ID (1)
        q_unk = "what elephant is"
        encoded_unk = encode_question(q_unk, self.vocab, max_len=self.seq_len)
        self.assertEqual(encoded_unk[1].item(), UNK_ID)

        # Test 3: Truncation for questions longer than max_len
        q_long = "word " * 35
        encoded_long = encode_question(q_long, self.vocab, max_len=self.seq_len)
        self.assertEqual(list(encoded_long.shape), [self.seq_len])

        # Test 4: Batch of questions
        batch_q = ["what color is the dog", "what is"]
        encoded_batch = encode_question(batch_q, self.vocab, max_len=self.seq_len)
        self.assertEqual(list(encoded_batch.shape), [2, self.seq_len])

        print(f"[PASS] Tokenizer contract test passed: output shape = {list(encoded_short.shape)}")

    def test_question_encoder_shape(self):
        """
        Tests QuestionEncoder forward pass:
        Input: [B, 20] -> Output: [B, 1024]
        """
        encoder = QuestionEncoder(
            vocab_size=self.vocab_size,
            embed_dim=300,
            hidden_dim=self.hidden_dim,
            padding_idx=PAD_ID
        )
        encoder.eval()

        # Dummy input tokens: LongTensor of shape [8, 20]
        dummy_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.batch_size, self.seq_len),
            dtype=torch.long
        )

        with torch.no_grad():
            output = encoder(dummy_ids)

        expected_shape = [self.batch_size, self.hidden_dim]
        self.assertEqual(
            list(output.shape),
            expected_shape,
            f"Expected QuestionEncoder output shape {expected_shape}, got {list(output.shape)}"
        )
        # Ensure values are within tanh range [-1.0, 1.0]
        self.assertTrue(torch.all(output >= -1.0) and torch.all(output <= 1.0))

        print(f"[PASS] QuestionEncoder shape test passed: input {list(dummy_ids.shape)} -> output {list(output.shape)}")

    def test_vqa_model_shape(self):
        """
        Tests VQAModel forward pass:
        Input: image_features [B, 1024], question_ids [B, 20] -> Output: [B, 3000]
        """
        model = VQAModel(
            vocab_size=self.vocab_size,
            image_dim=self.image_dim,
            text_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout_prob=0.5
        )
        model.eval()

        # Dummy pooled ResNet-50 features: [8, 1024]
        dummy_image_features = torch.randn(self.batch_size, self.image_dim, dtype=torch.float32)

        # Dummy question token IDs: [8, 20]
        dummy_question_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.batch_size, self.seq_len),
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(dummy_image_features, dummy_question_ids)

        expected_shape = [self.batch_size, self.num_classes]
        self.assertEqual(
            list(logits.shape),
            expected_shape,
            f"Expected VQAModel output shape {expected_shape}, got {list(logits.shape)}"
        )

        print(f"[PASS] VQAModel shape test passed: image {list(dummy_image_features.shape)} + question {list(dummy_question_ids.shape)} -> logits {list(logits.shape)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

