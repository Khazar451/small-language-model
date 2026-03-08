"""
Streaming dataset for large-scale language model training.

Supports loading from multiple local files and directories without loading
all data into memory. Compatible with .txt, .jsonl, .parquet, and .arrow
formats, with automatic format detection and fallback encoding handling.
"""

import glob as _glob
import json
import logging
import os
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".txt", ".jsonl", ".json", ".parquet", ".arrow"}


# ---------------------------------------------------------------------------
# Low-level file iterators
# ---------------------------------------------------------------------------

def _iter_text_file(path: str) -> Iterator[str]:
    """Yield non-empty lines from a plain text file.

    Falls back to Latin-1 encoding if UTF-8 decoding fails.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip():
                        yield line
            return
        except UnicodeDecodeError:
            continue
    logger.warning("Could not decode %s with utf-8 or latin-1; skipping.", path)


def _iter_jsonl_file(path: str, text_field: str = "text") -> Iterator[str]:
    """Yield text values from a JSON Lines file.

    Each line must be a JSON object containing *text_field*.
    Lines that cannot be parsed are silently skipped.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text and text.strip():
                    yield text
            except json.JSONDecodeError:
                continue


def _iter_parquet_file(path: str, text_column: str = "text") -> Iterator[str]:
    """Yield text values from a Parquet file.

    Requires pandas + pyarrow. Falls back to the first string column when
    *text_column* is not present.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "pandas is required for Parquet support. "
            "Install with: pip install pandas pyarrow"
        )
        return

    df = pd.read_parquet(path)
    if text_column not in df.columns:
        str_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not str_cols:
            logger.warning("No text column found in %s", path)
            return
        text_column = str_cols[0]
        logger.debug("Using column '%s' from %s", text_column, path)

    for text in df[text_column]:
        if isinstance(text, str) and text.strip():
            yield text


def _iter_arrow_file(path: str, text_column: str = "text") -> Iterator[str]:
    """Yield text values from an Arrow IPC file.

    Requires pyarrow. Falls back to the first string-typed column when
    *text_column* is not present.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "pyarrow is required for Arrow support. "
            "Install with: pip install pyarrow"
        )
        return

    reader = pa.ipc.open_file(path)
    table = reader.read_all()

    if text_column not in table.schema.names:
        for field in table.schema:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                text_column = field.name
                break
        else:
            logger.warning("No text column found in %s", path)
            return

    for text in table[text_column].to_pylist():
        if isinstance(text, str) and text.strip():
            yield text


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_files(
    paths: Union[str, List[str]],
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Discover all matching file paths from the given paths.

    Args:
        paths: A file path, directory path, glob pattern, or a list of these.
        recursive: Whether to recurse into subdirectories when *paths*
            contains directories.
        extensions: File extensions to include (e.g. ``['.txt', '.jsonl']``).
            Defaults to all :data:`SUPPORTED_FORMATS`.

    Returns:
        Sorted, deduplicated list of absolute file paths.
    """
    if extensions is None:
        extensions = list(SUPPORTED_FORMATS)
    ext_set = {ext.lower() for ext in extensions}

    if isinstance(paths, str):
        paths = [paths]

    result: List[str] = []
    for path in paths:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in ext_set:
                result.append(path)
        elif os.path.isdir(path):
            pattern = "**/*" if recursive else "*"
            for ext in ext_set:
                matched = _glob.glob(
                    os.path.join(path, pattern + ext), recursive=recursive
                )
                result.extend(matched)
        else:
            # Treat as a glob pattern
            for mp in _glob.glob(path, recursive=recursive):
                if os.path.splitext(mp)[1].lower() in ext_set:
                    result.append(mp)

    return sorted(set(os.path.abspath(p) for p in result))


# ---------------------------------------------------------------------------
# StreamingTextDataset
# ---------------------------------------------------------------------------

class StreamingTextDataset:
    """Streaming dataset for large-scale language model pre-training.

    Reads text from multiple local files or directories on-the-fly without
    loading the entire dataset into memory. Supports ``.txt``, ``.jsonl``,
    ``.parquet``, and ``.arrow`` formats with automatic detection.

    Args:
        paths: File path(s), directory path(s), glob patterns, or a list
            of any combination of the above.
        tokenizer: HuggingFace-compatible tokenizer instance.
        max_seq_length: Maximum sequence length for each chunk.
        stride: Sliding-window stride (defaults to *max_seq_length* for
            non-overlapping chunks).
        recursive: Whether to recurse into subdirectories.
        shuffle: Whether to shuffle the file order before each pass.
        shuffle_buffer_size: Buffer size for ``tf.data`` element-level
            shuffling.
        text_field: JSON key containing text in ``.jsonl`` files.
        text_column: Column name containing text in ``.parquet`` /
            ``.arrow`` files.
        extensions: File extensions to include (defaults to all supported
            formats).
        seed: Random seed for reproducible shuffling.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = StreamingTextDataset(
        ...     paths="data/",
        ...     tokenizer=tokenizer,
        ...     max_seq_length=1024,
        ...     recursive=True,
        ... )
        >>> tf_ds = dataset.get_tf_dataset(batch_size=8)
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        tokenizer: Any,
        max_seq_length: int = 1024,
        stride: Optional[int] = None,
        recursive: bool = False,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        text_field: str = "text",
        text_column: str = "text",
        extensions: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        self.recursive = recursive
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.text_field = text_field
        self.text_column = text_column
        self.seed = seed

        self.files = collect_files(paths, recursive=recursive, extensions=extensions)
        if not self.files:
            raise ValueError(
                f"No supported files found in: {paths}. "
                f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
            )
        logger.info("StreamingTextDataset: found %d files", len(self.files))

    def _iter_file(self, path: str) -> Iterator[str]:
        """Yield raw text strings from a single file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            yield from _iter_text_file(path)
        elif ext in {".json", ".jsonl"}:
            yield from _iter_jsonl_file(path, text_field=self.text_field)
        elif ext == ".parquet":
            yield from _iter_parquet_file(path, text_column=self.text_column)
        elif ext == ".arrow":
            yield from _iter_arrow_file(path, text_column=self.text_column)
        else:
            logger.warning("Unsupported file format, skipping: %s", path)

    def stream_texts(self) -> Iterator[str]:
        """Yield raw text strings from all source files.

        When *shuffle* is enabled the file order is randomised on each call.

        Yields:
            Non-empty text strings from all source files.
        """
        files = list(self.files)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(files)
        for path in files:
            logger.debug("Streaming from: %s", path)
            yield from self._iter_file(path)

    def _tokenize_and_chunk(self, text: str) -> Iterator[np.ndarray]:
        """Tokenize *text* and yield fixed-length token-ID chunks."""
        encoding = self.tokenizer(
            text,
            return_tensors="np",
            truncation=False,
            add_special_tokens=True,
        )
        all_ids = encoding["input_ids"][0]
        for start in range(0, len(all_ids), self.stride):
            end = min(start + self.max_seq_length, len(all_ids))
            chunk = all_ids[start:end].astype(np.int32)
            if len(chunk) >= 2:
                yield chunk

    def _generator(self) -> Iterator[tuple]:
        """Generator yielding ``(input_ids, attention_mask)`` pairs."""
        for text in self.stream_texts():
            for chunk in self._tokenize_and_chunk(text):
                mask = np.ones(len(chunk), dtype=np.int32)
                yield chunk, mask

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        repeat: bool = False,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        """Create a streaming ``tf.data.Dataset``.

        Args:
            batch_size: Number of examples per batch.
            repeat: Whether to repeat the dataset indefinitely.
            prefetch: Whether to prefetch batches asynchronously.

        Returns:
            ``tf.data.Dataset`` yielding ``(input_ids, attention_mask)``
            batches of shape ``(batch_size, seq_len)``.
        """
        pad_id = int(self.tokenizer.pad_token_id or 0)

        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
        )

        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(tf.cast(pad_id, tf.int32), tf.cast(0, tf.int32)),
        )

        if repeat:
            dataset = dataset.repeat()

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
