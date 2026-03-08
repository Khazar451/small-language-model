"""
HuggingFace Datasets integration for large-scale language model training.

Provides memory-efficient streaming access to datasets hosted on the
HuggingFace Hub without downloading them entirely to disk.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of popular large-scale pre-training datasets
# ---------------------------------------------------------------------------

RECOMMENDED_DATASETS: Dict[str, Dict] = {
    "openwebtext": {
        "name": "openwebtext",
        "split": "train",
        "text_field": "text",
        "size_gb": 40,
        "tokens_b": 8,
        "description": "High-quality web text (~8B tokens)",
    },
    "the_pile": {
        "name": "EleutherAI/pile",
        "split": "train",
        "text_field": "text",
        "size_gb": 825,
        "tokens_b": 210,
        "description": "Diverse high-quality text (~210B tokens)",
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "split": "train",
        "text_field": "text",
        "size_gb": 90,   # ~16 GiB compressed, ~90 GiB uncompressed
        "tokens_b": 627,
        "description": "Deduplicated web + code data (~627B tokens, ~90 GiB uncompressed)",
    },
    "fineweb": {
        "name": "HuggingFaceFW/fineweb",
        "split": "train",
        "text_field": "text",
        "size_gb": 5000,
        "tokens_b": 15000,
        "description": "Educational web data (~15T tokens)",
    },
    "falcon_refinedweb": {
        "name": "tiiuae/falcon-refinedweb",
        "split": "train",
        "text_field": "content",
        "size_gb": 2800,
        "tokens_b": 600,
        "description": "High-quality deduplicated web data (~600B tokens)",
    },
}


class HuggingFaceLoader:
    """Load and stream datasets from the HuggingFace Hub.

    Provides memory-efficient access to large datasets. When
    *streaming=True* (the default) the dataset is never fully
    downloaded to disk; individual records are fetched on demand.

    Args:
        dataset_name: HuggingFace dataset identifier
            (e.g. ``'openwebtext'``).  Short keys from
            :data:`RECOMMENDED_DATASETS` are also accepted.
        tokenizer: HuggingFace-compatible tokenizer instance.
        split: Dataset split to load (e.g. ``'train'``).
        subset: Dataset subset / configuration name.
        text_field: Column name containing the raw text.
        max_seq_length: Maximum sequence length for tokenized chunks.
        stride: Sliding-window stride (defaults to *max_seq_length*).
        streaming: Whether to use HuggingFace streaming mode.
        max_samples: Maximum number of documents to use.
        shuffle_buffer_size: Buffer size for element-level shuffling.
            Set to 0 to disable shuffling.
        cache_dir: Directory for caching downloaded data.
        trust_remote_code: Whether to allow remote code in dataset
            loading scripts.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> loader = HuggingFaceLoader(
        ...     dataset_name="openwebtext",
        ...     tokenizer=tokenizer,
        ...     streaming=True,
        ... )
        >>> tf_ds = loader.get_tf_dataset(batch_size=8)
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        split: str = "train",
        subset: Optional[str] = None,
        text_field: str = "text",
        max_seq_length: int = 1024,
        stride: Optional[int] = None,
        streaming: bool = True,
        max_samples: Optional[int] = None,
        shuffle_buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.subset = subset
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        self.streaming = streaming
        self.max_samples = max_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        # Resolve short key → full dataset name / text field
        if dataset_name in RECOMMENDED_DATASETS:
            info = RECOMMENDED_DATASETS[dataset_name]
            self.dataset_name = info["name"]
            self.text_field = info["text_field"] if text_field == "text" else text_field
            logger.info(
                "Loading known dataset '%s': %s",
                dataset_name,
                info["description"],
            )
        else:
            self.dataset_name = dataset_name
            self.text_field = text_field

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """Return the HuggingFace dataset object."""
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install with: pip install datasets"
            ) from exc

        load_kwargs: Dict[str, Any] = {
            "streaming": self.streaming,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        if self.subset:
            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **load_kwargs,
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                **load_kwargs,
            )

        if self.max_samples is not None and not self.streaming:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        return dataset

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_texts(self) -> Iterator[str]:
        """Stream raw text strings from the dataset.

        Yields:
            Non-empty text strings up to *max_samples* documents.
        """
        dataset = self._load_hf_dataset()
        count = 0
        for example in dataset:
            text = example.get(self.text_field, "")
            if text and text.strip():
                yield text
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    break

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        repeat: bool = False,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        """Create a streaming ``tf.data.Dataset`` from the HuggingFace dataset.

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

        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

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

    @staticmethod
    def list_recommended() -> Dict[str, Dict]:
        """Return the registry of recommended pre-training datasets.

        Returns:
            Mapping from short dataset keys to their metadata.
        """
        return dict(RECOMMENDED_DATASETS)
