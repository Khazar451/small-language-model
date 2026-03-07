"""
Tests for data loading and preprocessing utilities.

Tests cover:
- DataPreprocessor text cleaning
- DataPreprocessor text splitting
- TextDataset creation and iteration
- ClassificationDataset creation and iteration
- QADataset creation and iteration
- tokenize_dataset utility function
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor, tokenize_dataset


# ---------------------------------------------------------------------------
# DataPreprocessor tests (no tokenizer needed)
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    def setup_method(self):
        # Use a non-downloading config for preprocessing-only tests
        self.preprocessor = DataPreprocessor.__new__(DataPreprocessor)
        self.preprocessor.tokenizer_name = "gpt2"
        self.preprocessor.max_seq_length = 128
        self.preprocessor.lowercase = False
        self.preprocessor.remove_special_chars = False
        self.preprocessor._tokenizer = None

    def test_clean_text_whitespace(self):
        result = self.preprocessor.clean_text("Hello   world  ")
        assert result == "Hello world"

    def test_clean_text_null_bytes(self):
        result = self.preprocessor.clean_text("Hello\x00World")
        assert "\x00" not in result

    def test_clean_text_lowercase(self):
        preprocessor = DataPreprocessor.__new__(DataPreprocessor)
        preprocessor.lowercase = True
        preprocessor.remove_special_chars = False
        preprocessor._tokenizer = None
        result = preprocessor.clean_text("Hello WORLD")
        assert result == "hello world"

    def test_clean_text_empty(self):
        result = self.preprocessor.clean_text("   ")
        assert result == ""

    def test_clean_texts_filters_empty(self):
        texts = ["Hello", "   ", "World", ""]
        result = self.preprocessor.clean_texts(texts)
        assert len(result) == 2
        assert "Hello" in result
        assert "World" in result

    def test_split_dataset_ratios(self):
        texts = [f"text {i}" for i in range(100)]
        train, val, test = self.preprocessor.split_dataset(
            texts, ratios=(0.8, 0.1, 0.1), shuffle=False
        )
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_dataset_no_overlap(self):
        texts = [f"text_{i}" for i in range(50)]
        train, val, test = self.preprocessor.split_dataset(
            texts, ratios=(0.6, 0.2, 0.2), shuffle=False
        )
        all_texts = set(train + val + test)
        assert len(all_texts) == 50  # No duplicates

    def test_split_dataset_invalid_ratios(self):
        texts = ["a", "b", "c"]
        with pytest.raises(AssertionError):
            self.preprocessor.split_dataset(texts, ratios=(0.5, 0.5, 0.5))

    def test_split_dataset_shuffle(self):
        texts = [f"text_{i}" for i in range(100)]
        train1, _, _ = self.preprocessor.split_dataset(texts, shuffle=True, seed=42)
        train2, _, _ = self.preprocessor.split_dataset(texts, shuffle=True, seed=42)
        assert train1 == train2  # Same seed = same result

    def test_split_dataset_different_seeds(self):
        texts = [f"text_{i}" for i in range(100)]
        train1, _, _ = self.preprocessor.split_dataset(texts, shuffle=True, seed=42)
        train2, _, _ = self.preprocessor.split_dataset(texts, shuffle=True, seed=99)
        # Different seeds should give different orderings (with high probability)
        assert train1 != train2

    def test_save_splits(self):
        texts = [f"text {i}" for i in range(30)]
        train, val, test = self.preprocessor.split_dataset(
            texts, ratios=(0.8, 0.1, 0.1), shuffle=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self.preprocessor.save_splits(train, val, test, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "train.txt"))
            assert os.path.exists(os.path.join(tmpdir, "val.txt"))
            assert os.path.exists(os.path.join(tmpdir, "test.txt"))

            with open(os.path.join(tmpdir, "train.txt")) as f:
                loaded = f.read().split("\n")
            assert len(loaded) == len(train)


# ---------------------------------------------------------------------------
# TextDataset tests (uses offline simple tokenizer)
# ---------------------------------------------------------------------------

class TestTextDataset:
    def _write_temp_file(self, text: str):
        """Create a temporary text file."""
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmpfile.write(text)
        tmpfile.close()
        return tmpfile.name

    def test_basic_loading(self, simple_tokenizer):
        text = "Hello world. " * 50
        path = self._write_temp_file(text)
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(path, simple_tokenizer, max_seq_length=32)
            assert len(dataset) > 0
        finally:
            os.unlink(path)

    def test_chunk_size(self, simple_tokenizer):
        text = "word " * 500
        path = self._write_temp_file(text)
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(path, simple_tokenizer, max_seq_length=64)
            for ex in dataset.examples:
                assert len(ex["input_ids"]) <= 64
        finally:
            os.unlink(path)

    def test_tf_dataset_output(self, simple_tokenizer):
        text = "The quick brown fox. " * 100
        path = self._write_temp_file(text)
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(path, simple_tokenizer, max_seq_length=32)
            tf_ds = dataset.get_tf_dataset(batch_size=2, shuffle=False)
            for batch in tf_ds.take(1):
                input_ids, attention_mask = batch
                assert input_ids.shape[0] <= 2
                assert input_ids.dtype == tf.int32
        finally:
            os.unlink(path)

    def test_file_not_found(self, simple_tokenizer):
        from src.data.dataset import TextDataset
        with pytest.raises(FileNotFoundError):
            TextDataset("/nonexistent/path.txt", simple_tokenizer)

    def test_utf8_bom_encoding(self, simple_tokenizer):
        """TextDataset should load files encoded as UTF-8 with BOM (utf-8-sig)."""
        text = "Hello world. " * 50
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8-sig"
        )
        tmpfile.write(text)
        tmpfile.close()
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(tmpfile.name, simple_tokenizer, max_seq_length=32)
            assert len(dataset) > 0
        finally:
            os.unlink(tmpfile.name)

    def test_utf16_encoding(self, simple_tokenizer):
        """TextDataset should load files encoded as UTF-16."""
        text = "Hello world. " * 50
        tmpfile = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        tmpfile.write(text.encode("utf-16"))
        tmpfile.close()
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(tmpfile.name, simple_tokenizer, max_seq_length=32)
            assert len(dataset) > 0
        finally:
            os.unlink(tmpfile.name)

    def test_latin1_encoding(self, simple_tokenizer):
        """TextDataset should load files encoded as Latin-1."""
        text = "café naïve résumé. " * 50
        tmpfile = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        tmpfile.write(text.encode("latin-1"))
        tmpfile.close()
        try:
            from src.data.dataset import TextDataset
            dataset = TextDataset(tmpfile.name, simple_tokenizer, max_seq_length=32)
            assert len(dataset) > 0
        finally:
            os.unlink(tmpfile.name)

    def test_unsupported_encoding_raises(self, simple_tokenizer):
        """TextDataset should raise ValueError when all encodings fail."""
        from unittest.mock import patch
        from src.data.dataset import TextDataset

        # Create a real (empty) temp file so os.path.exists passes, then
        # patch open to always raise UnicodeDecodeError so every encoding fails.
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmpfile.write("x")
        tmpfile.close()
        try:
            with patch(
                "builtins.open",
                side_effect=UnicodeDecodeError("test", b"", 0, 1, "forced"),
            ):
                with pytest.raises(ValueError, match="Unable to decode"):
                    TextDataset(tmpfile.name, simple_tokenizer)
        finally:
            os.unlink(tmpfile.name)


# ---------------------------------------------------------------------------
# ClassificationDataset tests
# ---------------------------------------------------------------------------

class TestClassificationDataset:
    def test_basic_creation(self, simple_tokenizer):
        from src.data.dataset import ClassificationDataset
        texts = ["I love this!", "I hate this!", "It is okay."]
        labels = [1, 0, 0]
        dataset = ClassificationDataset(texts, labels, simple_tokenizer, max_seq_length=32)
        assert len(dataset) == 3

    def test_tf_dataset_output(self, simple_tokenizer):
        from src.data.dataset import ClassificationDataset
        texts = ["Good movie!", "Bad film.", "Average.", "Great!", "Terrible!"]
        labels = [1, 0, 0, 1, 0]
        dataset = ClassificationDataset(texts, labels, simple_tokenizer, max_seq_length=32)
        tf_ds = dataset.get_tf_dataset(batch_size=2, shuffle=False)

        for inputs, lbl in tf_ds.take(1):
            assert "input_ids" in inputs
            assert "attention_mask" in inputs
            assert inputs["input_ids"].dtype == tf.int32
            assert lbl.dtype == tf.int32

    def test_label_map(self, simple_tokenizer):
        from src.data.dataset import ClassificationDataset
        texts = ["positive text", "negative text"]
        labels = [1, 0]
        label_map = {"negative": 0, "positive": 1}
        dataset = ClassificationDataset(texts, labels, simple_tokenizer, label_map=label_map)
        assert len(dataset) == 2

    def test_mismatched_inputs_raises(self, simple_tokenizer):
        from src.data.dataset import ClassificationDataset
        with pytest.raises(AssertionError):
            ClassificationDataset(["a", "b"], [0], simple_tokenizer)

    def test_padding_to_max_length(self, simple_tokenizer):
        from src.data.dataset import ClassificationDataset
        texts = ["Short.", "This is a much longer text that has more tokens in it."]
        labels = [0, 1]
        dataset = ClassificationDataset(texts, labels, simple_tokenizer, max_seq_length=32)
        # All sequences should be padded to max_seq_length
        assert dataset.encodings["input_ids"].shape[1] == 32


# ---------------------------------------------------------------------------
# QADataset tests
# ---------------------------------------------------------------------------

class TestQADataset:
    def test_basic_creation(self, bert_like_tokenizer):
        from src.data.dataset import QADataset
        contexts = ["Paris is the capital of France."]
        questions = ["What is the capital of France?"]
        answers = [{"text": "Paris", "answer_start": 0}]
        dataset = QADataset(
            contexts, questions, answers, bert_like_tokenizer, max_seq_length=64
        )
        assert len(dataset) == 1

    def test_tf_dataset_output(self, bert_like_tokenizer):
        from src.data.dataset import QADataset
        contexts = [
            "The Amazon is the largest river.",
            "Python was created in 1991.",
        ]
        questions = [
            "What is the Amazon?",
            "When was Python created?",
        ]
        answers = [
            {"text": "largest river", "answer_start": 18},
            {"text": "1991", "answer_start": 22},
        ]
        dataset = QADataset(
            contexts, questions, answers, bert_like_tokenizer, max_seq_length=64
        )
        tf_ds = dataset.get_tf_dataset(batch_size=2)

        for batch in tf_ds.take(1):
            assert "input_ids" in batch
            assert "start_positions" in batch
            assert "end_positions" in batch


# ---------------------------------------------------------------------------
# tokenize_dataset utility tests
# ---------------------------------------------------------------------------

class TestTokenizeDataset:
    def test_basic_tokenization(self, simple_tokenizer):
        texts = ["Hello world", "Machine learning"]
        result = tokenize_dataset(texts, simple_tokenizer, max_length=32)
        assert "input_ids" in result
        assert result["input_ids"].shape == (2, 32)

    def test_attention_mask_present(self, simple_tokenizer):
        texts = ["Short.", "This is a longer text."]
        result = tokenize_dataset(texts, simple_tokenizer, max_length=16)
        assert "attention_mask" in result
