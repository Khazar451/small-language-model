"""
Streaming dataset for memory-efficient processing of large text corpora.

Supports terabyte-scale datasets by reading files lazily instead of loading
everything into memory, and integrates directly with tf.data pipelines.
"""

import glob as _glob
import logging
import os
from typing import Any, Iterator, List, Optional, Union

import tensorflow as tf

logger = logging.getLogger(__name__)


class StreamingDataset:
    """Memory-efficient streaming dataset for large text corpora.

    Reads source files lazily, tokenises on the fly, and yields fixed-length
    chunks suitable for language-model pre-training.  Any number of glob
    patterns or explicit file paths can be mixed as sources.

    Args:
        sources: List of file paths or glob patterns (e.g. ``"data/*.jsonl"``).
            Plain ``.txt`` files and JSON-Lines (``.jsonl``) files are supported.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Length of each output token-ID chunk.
        buffer_size: Number of examples to prefetch into the shuffle buffer.
        text_field: Key to use when reading JSON-Lines files (default ``"text"``).

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = StreamingDataset(
        ...     sources=["data/shard_*.jsonl"],
        ...     tokenizer=tokenizer,
        ...     max_seq_length=2048,
        ... )
        >>> train_ds = dataset.get_tf_dataset(batch_size=4, shuffle=True)
    """

    def __init__(
        self,
        sources: List[str],
        tokenizer: Any,
        max_seq_length: int = 2048,
        buffer_size: int = 10_000,
        text_field: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.text_field = text_field
        self.file_paths = self._resolve_sources(sources)
        if not self.file_paths:
            logger.warning("StreamingDataset: no files found for sources %s", sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sources(sources: List[str]) -> List[str]:
        """Expand glob patterns and return a sorted list of file paths."""
        paths: List[str] = []
        for pattern in sources:
            expanded = _glob.glob(pattern, recursive=True)
            if expanded:
                paths.extend(expanded)
            elif os.path.isfile(pattern):
                paths.append(pattern)
        return sorted(paths)

    def _iter_texts(self) -> Iterator[str]:
        """Yield raw text strings from all source files."""
        import json

        for path in self.file_paths:
            ext = os.path.splitext(path)[-1].lower()
            try:
                with open(path, encoding="utf-8") as fh:
                    if ext == ".jsonl":
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                yield obj.get(self.text_field, "")
                            except json.JSONDecodeError:
                                continue
                    else:
                        yield fh.read()
            except OSError as exc:
                logger.warning("StreamingDataset: could not read %s: %s", path, exc)

    def _iter_chunks(self) -> Iterator[List[int]]:
        """Tokenise texts and yield fixed-length token-ID chunks."""
        buffer: List[int] = []
        for text in self._iter_texts():
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= self.max_seq_length + 1:
                chunk = buffer[: self.max_seq_length + 1]
                buffer = buffer[self.max_seq_length:]
                yield chunk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Build a ``tf.data.Dataset`` ready for model training.

        Each element is a dict with keys ``"input_ids"`` and ``"labels"``
        (the labels are the input_ids shifted left by one position, as is
        standard for causal language-model training).

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle the dataset.

        Returns:
            A batched ``tf.data.Dataset``.
        """
        seq_len = self.max_seq_length

        def _generator():
            for chunk in self._iter_chunks():
                input_ids = chunk[:seq_len]
                labels = chunk[1: seq_len + 1]
                # Pad if shorter than expected (last chunk of a file)
                pad = seq_len - len(input_ids)
                if pad > 0:
                    pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                    input_ids = input_ids + [pad_id] * pad
                    labels = labels + [-100] * pad
                yield {"input_ids": input_ids, "labels": labels}

        output_signature = {
            "input_ids": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        }

        ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
