"""
Tokenization caching to avoid re-tokenizing data on every training run.

Tokenized chunks are persisted to disk as compressed NumPy arrays so that
subsequent training runs can skip the tokenization step entirely.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"


class TokenizerCache:
    """Cache tokenized data to disk to skip re-tokenization on subsequent runs.

    Tokenized chunks are stored as ``.npy`` files inside *cache_dir*.
    A ``metadata.json`` file records chunk file paths and statistics so that
    the cache can be validated and inspected without loading all the data.

    Tokenization is resumable: if a previous run was interrupted the cache
    can be rebuilt from scratch or appended to (controlled by *overwrite*).

    Args:
        cache_dir: Directory where cached data is stored.
        tokenizer: HuggingFace-compatible tokenizer instance.
        max_seq_length: Maximum sequence length for each chunk.
        stride: Sliding-window stride (defaults to *max_seq_length* for
            non-overlapping chunks).
        overwrite: If ``True``, existing cached data is deleted before
            tokenizing. If ``False`` (default), an existing cache is
            returned as-is without re-tokenizing.

    Example:
        >>> cache = TokenizerCache(
        ...     cache_dir=".cache/tokenized",
        ...     tokenizer=tokenizer,
        ...     max_seq_length=1024,
        ... )
        >>> cache.tokenize_texts(open("data/train.txt"))
        >>> tf_ds = cache.get_tf_dataset(batch_size=8)
    """

    def __init__(
        self,
        cache_dir: str,
        tokenizer: Any,
        max_seq_length: int = 1024,
        stride: Optional[int] = None,
        overwrite: bool = False,
    ):
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        self.overwrite = overwrite

        os.makedirs(cache_dir, exist_ok=True)

        self._metadata: Dict = {}
        self._chunk_paths: List[str] = []
        self._load_metadata()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @property
    def _metadata_path(self) -> str:
        return os.path.join(self.cache_dir, _METADATA_FILENAME)

    def _load_metadata(self) -> None:
        """Load metadata from disk if present."""
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path, "r") as f:
                self._metadata = json.load(f)
            self._chunk_paths = self._metadata.get("chunk_files", [])

    def _save_metadata(self) -> None:
        """Persist metadata to disk."""
        self._metadata["chunk_files"] = self._chunk_paths
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_cached(self) -> bool:
        """Return ``True`` when valid cached data exists on disk."""
        return bool(self._chunk_paths) and all(
            os.path.exists(p) for p in self._chunk_paths
        )

    def clear(self) -> None:
        """Delete all cached data from disk."""
        import shutil  # noqa: PLC0415

        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        self._metadata = {}
        self._chunk_paths = []

    def tokenize_texts(
        self,
        texts: Union[Iterator[str], List[str]],
        chunk_size: int = 10_000,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize an iterable of texts and save the result to disk.

        If :meth:`is_cached` returns ``True`` and *overwrite* is ``False``
        the method returns immediately with the cached statistics.

        Args:
            texts: Iterable of raw text strings.
            chunk_size: Number of sequences to store per ``.npy`` file.
            show_progress: Whether to log progress every 10 × *chunk_size*
                sequences.

        Returns:
            Statistics dictionary with token / chunk counts.
        """
        if self.is_cached() and not self.overwrite:
            logger.info(
                "Cache already exists at %s – skipping tokenization.", self.cache_dir
            )
            return self._metadata.get("stats", {})

        if self.overwrite:
            self.clear()

        pad_id = self.tokenizer.pad_token_id or 0
        total_chunks = 0
        total_tokens = 0
        file_index = 0
        current_buffer: List[np.ndarray] = []

        def _flush(buf: List[np.ndarray], idx: int) -> str:
            max_len = max(len(x) for x in buf)
            padded = np.stack(
                [np.pad(x, (0, max_len - len(x)), constant_values=pad_id) for x in buf]
            ).astype(np.int32)
            path = os.path.join(self.cache_dir, f"chunk_{idx:06d}.npy")
            np.save(path, padded)
            return path

        for text in texts:
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
                if len(chunk) < 2:
                    continue
                current_buffer.append(chunk)
                total_tokens += len(chunk)

                if len(current_buffer) >= chunk_size:
                    path = _flush(current_buffer, file_index)
                    self._chunk_paths.append(path)
                    file_index += 1
                    total_chunks += len(current_buffer)
                    current_buffer = []

                    if show_progress and total_chunks % (chunk_size * 10) == 0:
                        logger.info(
                            "Tokenized %d chunks (%d tokens so far).",
                            total_chunks,
                            total_tokens,
                        )

        if current_buffer:
            path = _flush(current_buffer, file_index)
            self._chunk_paths.append(path)
            total_chunks += len(current_buffer)

        stats: Dict[str, Any] = {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "num_files": len(self._chunk_paths),
            "max_seq_length": self.max_seq_length,
            "stride": self.stride,
        }
        self._metadata["stats"] = stats
        self._save_metadata()

        logger.info(
            "Tokenization complete: %d chunks, %d tokens in %d file(s).",
            total_chunks,
            total_tokens,
            len(self._chunk_paths),
        )
        return stats

    def _generator(self, shuffle: bool = False, seed: int = 42) -> Iterator[tuple]:
        """Yield ``(input_ids, attention_mask)`` pairs from cached files."""
        paths = list(self._chunk_paths)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(paths)

        for path in paths:
            if not os.path.exists(path):
                logger.warning("Cache file missing, skipping: %s", path)
                continue
            data = np.load(path)  # shape: (n, seq_len)
            indices = np.arange(len(data))
            if shuffle:
                rng.shuffle(indices)
            pad_id = self.tokenizer.pad_token_id or 0
            for i in indices:
                ids = data[i]
                mask = (ids != pad_id).astype(np.int32)
                yield ids, mask

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        repeat: bool = False,
        seed: int = 42,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        """Create a ``tf.data.Dataset`` from the cached tokenized data.

        :meth:`tokenize_texts` must be called first if the cache is empty.

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle within each chunk file.
            repeat: Whether to repeat the dataset indefinitely.
            seed: Random seed for shuffling.
            prefetch: Whether to prefetch batches asynchronously.

        Returns:
            ``tf.data.Dataset`` yielding ``(input_ids, attention_mask)``
            batches.

        Raises:
            RuntimeError: If no cached data is found.
        """
        if not self.is_cached():
            raise RuntimeError(
                f"No cached data found in '{self.cache_dir}'. "
                "Call tokenize_texts() first."
            )

        pad_id = int(self.tokenizer.pad_token_id or 0)

        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(shuffle=shuffle, seed=seed),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
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

    def get_stats(self) -> Dict[str, Any]:
        """Return the tokenization statistics recorded in the metadata."""
        return dict(self._metadata.get("stats", {}))
