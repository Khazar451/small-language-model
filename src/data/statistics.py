"""
Data statistics and monitoring for training data pipelines.

Analyses token distributions, vocabulary coverage, and document-level
quality metrics. Results can be saved to JSON for later inspection.
"""

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _compute_histogram(values: np.ndarray, num_bins: int = 20) -> Dict[str, list]:
    """Compute a simple histogram of numeric values.

    Args:
        values: 1-D array of numeric values.
        num_bins: Number of histogram bins.

    Returns:
        Dictionary with ``'bin_edges'`` and ``'counts'`` lists.
    """
    counts, bin_edges = np.histogram(values, bins=num_bins)
    return {
        "bin_edges": [float(x) for x in bin_edges],
        "counts": [int(x) for x in counts],
    }


class DataStatistics:
    """Compute and track statistics for training data.

    Analyses token-length distributions, vocabulary frequencies, and basic
    quality metrics. Results can be saved to JSON for offline inspection.

    Args:
        tokenizer: HuggingFace-compatible tokenizer instance.
        output_path: Optional file path to write statistics as JSON.
        max_vocab_items: Maximum number of vocabulary entries to track.

    Example:
        >>> stats = DataStatistics(tokenizer, output_path="data/statistics.json")
        >>> stats.analyze_texts(open("data/train.txt"))
        >>> stats.save()
    """

    def __init__(
        self,
        tokenizer: Any,
        output_path: Optional[str] = None,
        max_vocab_items: int = 10_000,
    ):
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.max_vocab_items = max_vocab_items
        self._stats: Dict[str, Any] = {}

    def analyze_texts(
        self,
        texts: Iterator[str],
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyse an iterable of raw text strings.

        Tokenizes each text (without truncation) and collects length
        distributions and vocabulary frequencies.

        Args:
            texts: Iterable of text strings.
            sample_size: If set, only the first *sample_size* non-empty
                texts are analysed.

        Returns:
            Dictionary of statistics, also stored in :attr:`_stats`.
        """
        lengths: List[int] = []
        vocab_counter: Counter = Counter()
        num_texts = 0
        num_empty = 0

        for text in texts:
            if sample_size is not None and num_texts >= sample_size:
                break

            if not text or not text.strip():
                num_empty += 1
                continue

            encoding = self.tokenizer(
                text, truncation=False, add_special_tokens=True,
                return_tensors="np",
            )
            ids = encoding["input_ids"][0].tolist()
            lengths.append(len(ids))
            vocab_counter.update(ids)
            num_texts += 1

        if not lengths:
            logger.warning("No texts analysed.")
            return {}

        lengths_arr = np.array(lengths)
        total_tokens = int(np.sum(lengths_arr))

        self._stats = {
            "num_texts": num_texts,
            "num_empty": num_empty,
            "total_tokens": total_tokens,
            "tokens_per_text": {
                "mean": float(np.mean(lengths_arr)),
                "std": float(np.std(lengths_arr)),
                "min": int(np.min(lengths_arr)),
                "max": int(np.max(lengths_arr)),
                "median": float(np.median(lengths_arr)),
                "p25": float(np.percentile(lengths_arr, 25)),
                "p75": float(np.percentile(lengths_arr, 75)),
                "p95": float(np.percentile(lengths_arr, 95)),
                "p99": float(np.percentile(lengths_arr, 99)),
            },
            "vocabulary": {
                "unique_tokens_seen": len(vocab_counter),
                "top_tokens": [
                    {"token_id": int(tok), "count": int(cnt)}
                    for tok, cnt in vocab_counter.most_common(100)
                ],
            },
            "length_histogram": _compute_histogram(lengths_arr),
        }

        logger.info(
            "Analysed %d texts: %d total tokens, %d unique tokens.",
            num_texts,
            total_tokens,
            len(vocab_counter),
        )
        return self._stats

    def analyze_token_sequences(
        self,
        sequences: Iterator[np.ndarray],
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyse pre-tokenized sequences.

        Useful for inspecting cached token ID arrays directly without
        decoding back to text.

        Args:
            sequences: Iterable of NumPy token-ID arrays.
            sample_size: If set, only the first *sample_size* sequences
                are analysed.

        Returns:
            Dictionary of statistics.
        """
        lengths: List[int] = []
        vocab_counter: Counter = Counter()
        num_sequences = 0

        for seq in sequences:
            if sample_size is not None and num_sequences >= sample_size:
                break
            lengths.append(len(seq))
            vocab_counter.update(seq.tolist())
            num_sequences += 1

        if not lengths:
            return {}

        lengths_arr = np.array(lengths)
        self._stats = {
            "num_sequences": num_sequences,
            "total_tokens": int(np.sum(lengths_arr)),
            "sequence_length": {
                "mean": float(np.mean(lengths_arr)),
                "std": float(np.std(lengths_arr)),
                "min": int(np.min(lengths_arr)),
                "max": int(np.max(lengths_arr)),
                "median": float(np.median(lengths_arr)),
            },
            "vocabulary": {
                "unique_tokens_seen": len(vocab_counter),
            },
        }
        return self._stats

    def save(self, path: Optional[str] = None) -> None:
        """Write statistics to a JSON file.

        Args:
            path: Destination file path. Falls back to *output_path* from
                ``__init__``.

        Raises:
            ValueError: If neither *path* nor *output_path* was provided.
        """
        target = path or self.output_path
        if target is None:
            raise ValueError(
                "No output path specified. Pass a path to save() or set "
                "output_path in the constructor."
            )
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w") as f:
            json.dump(self._stats, f, indent=2)
        logger.info("Statistics saved to %s.", target)

    def get_stats(self) -> Dict[str, Any]:
        """Return the last computed statistics dictionary."""
        return dict(self._stats)
