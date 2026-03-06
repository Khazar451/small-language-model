"""
Data preprocessing utilities.

This module provides utilities for cleaning, tokenizing, and preparing
text data for training language models.
"""

import logging
import re
import os
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Utilities for preprocessing raw text data.

    Handles text cleaning, normalization, and splitting into train/val/test sets.

    Args:
        tokenizer_name: Name of the HuggingFace tokenizer to use.
        max_seq_length: Maximum sequence length.
        lowercase: Whether to convert text to lowercase.
        remove_special_chars: Whether to remove special characters.

    Example:
        >>> preprocessor = DataPreprocessor("gpt2", max_seq_length=512)
        >>> cleaned = preprocessor.clean_text("Hello,   world!")
        >>> train, val, test = preprocessor.split_dataset(texts, ratios=(0.8, 0.1, 0.1))
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        lowercase: bool = False,
        remove_special_chars: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError as exc:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                ) from exc
        return self._tokenizer

    def clean_text(self, text: str) -> str:
        """Clean and normalize a single text string.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text string.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)

        # Remove null bytes and control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text

    def clean_texts(self, texts: List[str]) -> List[str]:
        """Clean a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of cleaned texts.
        """
        return [self.clean_text(t) for t in texts if t.strip()]

    def split_dataset(
        self,
        texts: List[str],
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split texts into train, validation, and test sets.

        Args:
            texts: List of text strings.
            ratios: Tuple of (train, val, test) split ratios. Must sum to 1.
            shuffle: Whether to shuffle before splitting.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_texts, val_texts, test_texts).
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

        if shuffle:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(texts))
            texts = [texts[i] for i in indices]

        n = len(texts)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        return texts[:train_end], texts[train_end:val_end], texts[val_end:]

    def save_splits(
        self,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        output_dir: str,
    ) -> None:
        """Save train/val/test splits to text files.

        Args:
            train_texts: Training texts.
            val_texts: Validation texts.
            test_texts: Test texts.
            output_dir: Directory to save files in.
        """
        os.makedirs(output_dir, exist_ok=True)

        for split_name, split_data in [
            ("train.txt", train_texts),
            ("val.txt", val_texts),
            ("test.txt", test_texts),
        ]:
            path = os.path.join(output_dir, split_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(split_data))
            logger.info("Saved %d examples to %s", len(split_data), path)

    def compute_token_statistics(
        self,
        texts: List[str],
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute tokenization statistics for a list of texts.

        Args:
            texts: List of text strings.
            sample_size: If provided, only compute stats on this many samples.

        Returns:
            Dictionary with statistics: mean/max/min token counts, etc.
        """
        if sample_size:
            rng = np.random.default_rng(42)
            texts = rng.choice(texts, min(sample_size, len(texts)), replace=False).tolist()

        lengths = []
        for text in texts:
            tokens = self.tokenizer(text, truncation=False)["input_ids"]
            lengths.append(len(tokens))

        lengths = np.array(lengths)
        return {
            "num_samples": len(texts),
            "mean_tokens": float(np.mean(lengths)),
            "std_tokens": float(np.std(lengths)),
            "min_tokens": int(np.min(lengths)),
            "max_tokens": int(np.max(lengths)),
            "median_tokens": float(np.median(lengths)),
            "pct_truncated": float(np.mean(lengths > self.max_seq_length)),
        }

    def prepare_from_hf_dataset(
        self,
        dataset_name: str,
        text_column: str = "text",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Load and prepare data from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'wikitext').
            text_column: Name of the column containing text.
            split: Dataset split to load.
            max_samples: Maximum number of samples to load.

        Returns:
            List of cleaned text strings.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required. Install with: pip install datasets"
            ) from exc

        logger.info("Loading dataset '%s' split '%s'", dataset_name, split)
        dataset = load_dataset(dataset_name, split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        texts = dataset[text_column]
        cleaned = self.clean_texts(texts)
        logger.info("Prepared %d texts from HuggingFace dataset", len(cleaned))
        return cleaned


def tokenize_dataset(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "np",
) -> Dict[str, np.ndarray]:
    """Tokenize a list of texts using a HuggingFace tokenizer.

    Args:
        texts: List of input texts.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        padding: Padding strategy.
        truncation: Whether to truncate long sequences.
        return_tensors: Return format ("np", "tf", or "pt").

    Returns:
        Dictionary with 'input_ids', 'attention_mask', etc.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
