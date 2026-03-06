"""
Pytest configuration and shared fixtures.

Provides a simple offline tokenizer for tests that don't require
network access to download pre-trained tokenizers.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleTokenizer:
    """A minimal tokenizer that works without network access.

    Character-level tokenizer for testing purposes. Supports the
    same interface as HuggingFace tokenizers.
    """

    def __init__(self, vocab_size: int = 256, max_length: int = 512):
        self.vocab_size = vocab_size
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 1
        self.unk_token = "[UNK]"
        self.unk_token_id = 2
        self.cls_token = "[CLS]"
        self.cls_token_id = 3
        self.sep_token = "[SEP]"
        self.sep_token_id = 4
        self.model_max_length = max_length
        self._special_ids = {0, 1, 2, 3, 4}

    def _text_to_ids(self, text: str) -> list:
        """Convert text to character-level token IDs."""
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        # Avoid special token IDs
        ids = [i if i not in self._special_ids else i + 10 for i in ids]
        return ids

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=None,
        padding=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=True,
        return_offsets_mapping=False,
        **kwargs,
    ):
        """Tokenize a single text, list of texts, or (text, text_pair) pair."""
        max_len = max_length or self.model_max_length

        # Detect Q&A style call: tokenizer(question, context, ...)
        if text_pair is not None:
            # Handle (question, context) pair
            ids_q = self._text_to_ids(str(text))
            ids_c = self._text_to_ids(str(text_pair))
            ids = ([self.cls_token_id] + ids_q[:max_len // 4] +
                   [self.sep_token_id] + ids_c)
            ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            ids_array = np.array([ids], dtype=np.int32)
            mask_array = np.array([mask], dtype=np.int32)

            offsets = [[i, i + 1] for i in range(len(ids))]
            offset_array = np.array([offsets], dtype=np.int32)

            result = {"input_ids": ids_array, "attention_mask": mask_array}
            if return_offsets_mapping:
                result["offset_mapping"] = offset_array

            if return_tensors == "tf":
                import tensorflow as tf
                result = {k: tf.constant(v) for k, v in result.items()}

            class Encoding(dict):
                def sequence_ids(self, batch_index=0):
                    half = len(self["input_ids"][batch_index]) // 2
                    return [0] * half + [1] * (len(self["input_ids"][batch_index]) - half)

                def pop(self, key, *args):
                    return dict.pop(self, key, *args)

            return Encoding(result)

        # Standard single/batch text tokenization
        single = isinstance(text, str)
        texts = [text] if single else list(text)

        batch_ids = []
        batch_mask = []

        for t in texts:
            ids = self._text_to_ids(t) if isinstance(t, str) else []
            if truncation:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)

        # Pad to same length if using padding (but not max_length)
        if padding and padding != "max_length":
            max_actual = max(len(ids) for ids in batch_ids) if batch_ids else 0
            for i in range(len(batch_ids)):
                pad_len = max_actual - len(batch_ids[i])
                batch_ids[i] = batch_ids[i] + [self.pad_token_id] * pad_len
                batch_mask[i] = batch_mask[i] + [0] * pad_len

        ids_array = np.array(batch_ids, dtype=np.int32)
        mask_array = np.array(batch_mask, dtype=np.int32)

        result = {"input_ids": ids_array, "attention_mask": mask_array}

        if return_offsets_mapping:
            offsets = []
            for t in texts:
                t_ids = self._text_to_ids(t) if isinstance(t, str) else []
                t_ids = t_ids[:max_len]
                offsets.append([[i, i + 1] for i in range(len(t_ids))])
            result["offset_mapping"] = np.array(offsets, dtype=np.int32)

        if return_tensors == "tf":
            import tensorflow as tf
            result = {k: tf.constant(v) for k, v in result.items()}

        class Encoding(dict):
            def sequence_ids(self, batch_index=0):
                half = len(self["input_ids"][batch_index]) // 2
                return [0] * half + [1] * (len(self["input_ids"][batch_index]) - half)

            def pop(self, key, *args):
                return dict.pop(self, key, *args)

        return Encoding(result)

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        return self._text_to_ids(text)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for i in ids:
            if skip_special_tokens and i in self._special_ids:
                continue
            if 32 <= i < 127:
                chars.append(chr(i))
        return "".join(chars)

    def batch_decode(self, ids_list, skip_special_tokens: bool = True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def save_pretrained(self, path: str):
        """No-op for testing."""
        pass


@pytest.fixture(scope="session")
def simple_tokenizer():
    """Provide a minimal offline tokenizer for tests."""
    return SimpleTokenizer(vocab_size=256, max_length=128)


@pytest.fixture(scope="session")
def gpt2_like_tokenizer():
    """Provide a GPT-2-like tokenizer for tests (uses SimpleTokenizer offline)."""
    tok = SimpleTokenizer(vocab_size=256, max_length=512)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="session")
def bert_like_tokenizer():
    """Provide a BERT-like tokenizer for tests (uses SimpleTokenizer offline)."""
    return SimpleTokenizer(vocab_size=256, max_length=384)
