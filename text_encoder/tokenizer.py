"""
Tokenizer module for Visual Question Answering (VQA).

This module handles text tokenization and integer ID encoding for questions.

Contract:
- Lowercase the question string.
- Whitespace split into words.
- Map words to integer IDs using a vocabulary dictionary:
    <pad> -> 0
    <unk> -> 1
    known words -> id >= 2
- Truncate or pad to max_len (default 20).
- Output shape: LongTensor of shape [20] for a single question,
  or [B, 20] if a list of questions is passed.
"""

from typing import Union, List, Dict
import torch

# Reserved special tokens and their assigned integer IDs
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize raw text by lowercasing and splitting on whitespace.
    Strictly follows the project contract: lowercased, whitespace split.

    Args:
        text (str): Raw question string (e.g., "What color is the cat?").

    Returns:
        List[str]: List of lowercased tokens.
    """
    if not isinstance(text, str):
        text = str(text)
    # Lowercase and split purely on whitespace
    return text.lower().strip().split()


def encode_question(
    question: Union[str, List[str]],
    vocab: Dict[str, int],
    max_len: int = 20
) -> torch.LongTensor:
    """
    Encodes one or more question strings into a PyTorch LongTensor of token IDs.

    Args:
        question: A single string or a list of question strings.
        vocab: Dictionary mapping word string -> integer ID (e.g., {"<pad>": 0, "<unk>": 1, ...}).
        max_len: Desired sequence length (default: 20).

    Returns:
        torch.LongTensor:
            - Shape [max_len] if a single question string is provided.
            - Shape [B, max_len] if a list of B question strings is provided.
    """
    # If a batch of questions is passed, encode each and stack into [B, max_len]
    if isinstance(question, (list, tuple)):
        return torch.stack([encode_question(q, vocab, max_len=max_len) for q in question], dim=0)

    # 1. Lowercase and split on whitespace
    tokens = tokenize_text(question)

    # 2. Look up special token IDs from vocab (falling back to constants if not found)
    pad_id = vocab.get(PAD_TOKEN, PAD_ID)
    unk_id = vocab.get(UNK_TOKEN, UNK_ID)

    # 3. Map tokens to IDs; unknown words are mapped to <unk> (ID 1)
    token_ids = [vocab.get(token, unk_id) for token in tokens]

    # 4. Truncate if token length exceeds max_len
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    # 5. Pad with <pad> (ID 0) up to max_len if shorter
    else:
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))

    # 6. Return as PyTorch LongTensor of shape [max_len]
    return torch.tensor(token_ids, dtype=torch.long)

