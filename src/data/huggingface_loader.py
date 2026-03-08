"""
HuggingFace Datasets integration for multi-source data loading.

Provides a convenient wrapper around the ``datasets`` library that supports
sampling from multiple datasets with configurable mixture weights and
seamless integration with the tf.data pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


class HuggingFaceDataLoader:
    """Load and interleave multiple HuggingFace datasets with mixture weights.

    Supports streaming mode (no full download required) and produces a
    ``tf.data.Dataset`` of fixed-length token-ID chunks ready for language
    model training.

    Args:
        datasets: List of dataset names accepted by ``datasets.load_dataset``
            (e.g. ``["wikitext", "openwebtext"]``).
        weights: Sampling probabilities for each dataset.  Must sum to 1 and
            have the same length as *datasets*.  Defaults to uniform weights.
        split: Dataset split to use (default ``"train"``).
        streaming: Whether to stream data without downloading (default ``True``).
        text_field: Column name that holds the raw text (default ``"text"``).
        dataset_configs: Optional mapping from dataset name to its config string
            (e.g. ``{"wikitext": "wikitext-103-raw-v1"}``).
        trust_remote_code: Passed directly to ``datasets.load_dataset``.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> loader = HuggingFaceDataLoader(
        ...     datasets=["wikitext", "openwebtext"],
        ...     weights=[0.4, 0.6],
        ...     streaming=True,
        ... )
        >>> train_ds = loader.get_interleaved_dataset(tokenizer, batch_size=8)
    """

    def __init__(
        self,
        datasets: List[str],
        weights: Optional[List[float]] = None,
        split: str = "train",
        streaming: bool = True,
        text_field: str = "text",
        dataset_configs: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
    ):
        if weights is not None and len(weights) != len(datasets):
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(datasets)={len(datasets)}"
            )
        self.dataset_names = datasets
        self.weights = weights or [1.0 / len(datasets)] * len(datasets)
        self.split = split
        self.streaming = streaming
        self.text_field = text_field
        self.dataset_configs = dataset_configs or {}
        self.trust_remote_code = trust_remote_code

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_datasets(self):
        """Load and return a list of HuggingFace dataset objects."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            ) from exc

        loaded = []
        for name in self.dataset_names:
            cfg = self.dataset_configs.get(name)
            kwargs: Dict[str, Any] = {
                "split": self.split,
                "streaming": self.streaming,
                "trust_remote_code": self.trust_remote_code,
            }
            if cfg:
                ds = load_dataset(name, cfg, **kwargs)
            else:
                ds = load_dataset(name, **kwargs)
            loaded.append(ds)
            logger.info("Loaded dataset '%s' (split=%s, streaming=%s)", name, self.split, self.streaming)
        return loaded

    @staticmethod
    def _interleave(hf_datasets, weights, text_field: str, seed: int = 42):
        """Interleave multiple HuggingFace datasets using the given weights."""
        try:
            from datasets import interleave_datasets  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            ) from exc

        return interleave_datasets(
            hf_datasets,
            probabilities=weights,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    def _iter_texts(self, hf_datasets, weights):
        """Yield raw text strings from the interleaved dataset."""
        merged = self._interleave(hf_datasets, weights, self.text_field)
        for example in merged:
            text = example.get(self.text_field, "")
            if text and text.strip():
                yield text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_interleaved_dataset(
        self,
        tokenizer: Any,
        batch_size: int = 8,
        max_seq_length: int = 2048,
        shuffle_buffer: int = 10_000,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Return a batched ``tf.data.Dataset`` from the interleaved sources.

        Args:
            tokenizer: HuggingFace tokenizer instance.
            batch_size: Number of examples per batch.
            max_seq_length: Token chunk length.
            shuffle_buffer: Shuffle buffer size (used only when *shuffle* is ``True``).
            shuffle: Whether to shuffle the dataset.

        Returns:
            A batched ``tf.data.Dataset`` with keys ``"input_ids"`` and ``"labels"``.
        """
        hf_datasets = self._load_hf_datasets()
        weights = self.weights
        text_field = self.text_field
        seq_len = max_seq_length

        def _generator():
            buffer = []
            for text in self._iter_texts(hf_datasets, weights):
                ids = tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(ids)
                while len(buffer) >= seq_len + 1:
                    chunk = buffer[: seq_len + 1]
                    buffer = buffer[seq_len:]
                    input_ids = chunk[:seq_len]
                    labels = chunk[1:]
                    yield {"input_ids": input_ids, "labels": labels}

        output_signature = {
            "input_ids": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        }

        ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
