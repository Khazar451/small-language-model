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


# ---------------------------------------------------------------------------
# StreamingTextDataset tests
# ---------------------------------------------------------------------------

class TestStreamingTextDataset:
    def _write_temp_txt(self, lines, suffix=".txt"):
        import tempfile
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        f.write("\n".join(lines))
        f.close()
        return f.name

    def _write_temp_jsonl(self, texts, field="text"):
        import json
        import tempfile
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for t in texts:
            f.write(json.dumps({field: t}) + "\n")
        f.close()
        return f.name

    def test_single_txt_file(self, simple_tokenizer):
        from src.data.streaming_dataset import StreamingTextDataset
        path = self._write_temp_txt(["Hello world"] * 20)
        try:
            ds = StreamingTextDataset(path, simple_tokenizer, max_seq_length=32, shuffle=False)
            texts = list(ds.stream_texts())
            assert len(texts) > 0
        finally:
            os.unlink(path)

    def test_jsonl_file(self, simple_tokenizer):
        from src.data.streaming_dataset import StreamingTextDataset
        path = self._write_temp_jsonl(["Hello world"] * 10)
        try:
            ds = StreamingTextDataset(path, simple_tokenizer, max_seq_length=32, shuffle=False)
            texts = list(ds.stream_texts())
            assert len(texts) == 10
        finally:
            os.unlink(path)

    def test_directory(self, simple_tokenizer):
        from src.data.streaming_dataset import StreamingTextDataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"file{i}.txt"), "w") as f:
                    f.write("\n".join([f"text line {j}" for j in range(10)]))
            ds = StreamingTextDataset(tmpdir, simple_tokenizer, max_seq_length=32, shuffle=False)
            texts = list(ds.stream_texts())
            assert len(texts) == 30

    def test_no_files_raises(self, simple_tokenizer):
        from src.data.streaming_dataset import StreamingTextDataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                StreamingTextDataset(tmpdir, simple_tokenizer)

    def test_tf_dataset_output(self, simple_tokenizer):
        from src.data.streaming_dataset import StreamingTextDataset
        path = self._write_temp_txt(["word " * 50] * 30)
        try:
            ds = StreamingTextDataset(
                path, simple_tokenizer, max_seq_length=32, shuffle=False
            )
            tf_ds = ds.get_tf_dataset(batch_size=4)
            for batch in tf_ds.take(1):
                ids, mask = batch
                assert ids.dtype == tf.int32
                assert mask.dtype == tf.int32
                assert ids.shape[0] <= 4
        finally:
            os.unlink(path)

    def test_collect_files(self, simple_tokenizer):
        from src.data.streaming_dataset import collect_files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.txt", "b.jsonl", "c.md"]:
                open(os.path.join(tmpdir, name), "w").close()
            files = collect_files(tmpdir, extensions=[".txt", ".jsonl"])
            assert len(files) == 2


# ---------------------------------------------------------------------------
# TokenizerCache tests
# ---------------------------------------------------------------------------

class TestTokenizerCache:
    def test_tokenize_and_load(self, simple_tokenizer):
        from src.data.tokenizer_cache import TokenizerCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizerCache(tmpdir, simple_tokenizer, max_seq_length=32)
            texts = ["Hello world. " * 10] * 20
            stats = cache.tokenize_texts(iter(texts), chunk_size=5)
            assert stats["total_chunks"] > 0
            assert stats["total_tokens"] > 0
            assert cache.is_cached()

    def test_skip_if_cached(self, simple_tokenizer):
        from src.data.tokenizer_cache import TokenizerCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizerCache(tmpdir, simple_tokenizer, max_seq_length=32)
            texts = ["Hello world. " * 5] * 10
            cache.tokenize_texts(iter(texts))
            # Second call should be a no-op
            stats2 = cache.tokenize_texts(iter(["new text"] * 100))
            # Should still return old stats
            assert stats2["total_chunks"] == cache.get_stats()["total_chunks"]

    def test_overwrite(self, simple_tokenizer):
        from src.data.tokenizer_cache import TokenizerCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizerCache(tmpdir, simple_tokenizer, max_seq_length=32)
            cache.tokenize_texts(iter(["Hello world. " * 5] * 10))
            first_count = cache.get_stats()["total_chunks"]

            cache2 = TokenizerCache(tmpdir, simple_tokenizer, max_seq_length=32, overwrite=True)
            cache2.tokenize_texts(iter(["Hello world. " * 5] * 20))
            assert cache2.get_stats()["total_chunks"] > first_count

    def test_get_tf_dataset(self, simple_tokenizer):
        from src.data.tokenizer_cache import TokenizerCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizerCache(tmpdir, simple_tokenizer, max_seq_length=32)
            texts = ["Hello world. " * 10] * 20
            cache.tokenize_texts(iter(texts), chunk_size=5)
            tf_ds = cache.get_tf_dataset(batch_size=4, shuffle=False)
            for batch in tf_ds.take(1):
                ids, mask = batch
                assert ids.dtype == tf.int32

    def test_get_tf_dataset_no_cache_raises(self, simple_tokenizer):
        from src.data.tokenizer_cache import TokenizerCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizerCache(tmpdir, simple_tokenizer)
            with pytest.raises(RuntimeError):
                cache.get_tf_dataset()


# ---------------------------------------------------------------------------
# DataStatistics tests
# ---------------------------------------------------------------------------

class TestDataStatistics:
    def test_basic_analysis(self, simple_tokenizer):
        from src.data.statistics import DataStatistics
        texts = ["Hello world"] * 10 + ["Machine learning is great"] * 5
        stats_obj = DataStatistics(simple_tokenizer)
        result = stats_obj.analyze_texts(iter(texts))
        assert result["num_texts"] == 15
        assert result["total_tokens"] > 0
        assert "tokens_per_text" in result
        assert "vocabulary" in result

    def test_empty_texts_counted(self, simple_tokenizer):
        from src.data.statistics import DataStatistics
        texts = ["Hello", "", "  ", "World"]
        stats_obj = DataStatistics(simple_tokenizer)
        result = stats_obj.analyze_texts(iter(texts))
        assert result["num_texts"] == 2
        assert result["num_empty"] == 2

    def test_sample_size(self, simple_tokenizer):
        from src.data.statistics import DataStatistics
        texts = [f"text number {i}" for i in range(100)]
        stats_obj = DataStatistics(simple_tokenizer)
        result = stats_obj.analyze_texts(iter(texts), sample_size=10)
        assert result["num_texts"] == 10

    def test_save_and_load(self, simple_tokenizer):
        from src.data.statistics import DataStatistics
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "stats.json")
            stats_obj = DataStatistics(simple_tokenizer, output_path=out)
            stats_obj.analyze_texts(iter(["Hello world"] * 5))
            stats_obj.save()
            assert os.path.exists(out)
            import json as _json
            with open(out) as f:
                data = _json.load(f)
            assert "total_tokens" in data

    def test_save_no_path_raises(self, simple_tokenizer):
        from src.data.statistics import DataStatistics
        stats_obj = DataStatistics(simple_tokenizer)
        stats_obj.analyze_texts(iter(["hello"] * 3))
        with pytest.raises(ValueError):
            stats_obj.save()


# ---------------------------------------------------------------------------
# MultiFileTextDataset tests
# ---------------------------------------------------------------------------

class TestMultiFileTextDataset:
    def test_basic_multi_file(self, simple_tokenizer):
        from src.data.dataset import MultiFileTextDataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                with open(os.path.join(tmpdir, f"f{i}.txt"), "w") as f:
                    f.write("\n".join([f"sample text line {j}" for j in range(20)]))
            ds = MultiFileTextDataset(tmpdir, simple_tokenizer, max_seq_length=32, shuffle=False)
            assert len(ds.files) == 3

    def test_tf_dataset_output(self, simple_tokenizer):
        from src.data.dataset import MultiFileTextDataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.txt"), "w") as f:
                f.write("\n".join(["word " * 50] * 30))
            ds = MultiFileTextDataset(tmpdir, simple_tokenizer, max_seq_length=32, shuffle=False)
            tf_ds = ds.get_tf_dataset(batch_size=4, shuffle=False)
            for batch in tf_ds.take(1):
                ids, mask = batch
                assert ids.dtype == tf.int32


# ---------------------------------------------------------------------------
# DataPreprocessor extended tests (HTML/URL cleaning, deduplication, filtering)
# ---------------------------------------------------------------------------

class TestDataPreprocessorExtended:
    def setup_method(self):
        from src.data.preprocessing import DataPreprocessor
        self.preprocessor = DataPreprocessor.__new__(DataPreprocessor)
        self.preprocessor.tokenizer_name = "gpt2"
        self.preprocessor.max_seq_length = 128
        self.preprocessor.lowercase = False
        self.preprocessor.remove_special_chars = False
        self.preprocessor._tokenizer = None

    def test_html_removal(self):
        result = self.preprocessor.clean_text("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result
        assert "Hello" in result
        assert "world" in result

    def test_url_removal(self):
        result = self.preprocessor.clean_text("Visit https://example.com for more.")
        assert "https://" not in result

    def test_email_removal(self):
        result = self.preprocessor.clean_text("Contact user@example.com for help.")
        assert "@" not in result

    def test_filter_by_length(self):
        texts = ["hi", "a" * 100, "b" * 200]
        result = self.preprocessor.filter_by_length(texts, min_length=10, max_length=150)
        assert len(result) == 1
        assert result[0] == "a" * 100

    def test_deduplicate_exact(self):
        texts = ["hello world", "hello world", "unique text"]
        result = self.preprocessor.deduplicate(texts)
        assert len(result) == 2
        assert "hello world" in result
        assert "unique text" in result

    def test_deduplicate_preserves_order(self):
        texts = ["a", "b", "a", "c", "b"]
        result = self.preprocessor.deduplicate(texts)
        assert result == ["a", "b", "c"]
