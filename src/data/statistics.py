"""
Dataset statistics and analysis utilities.

Computes and reports descriptive statistics for text datasets, including
token length distributions, vocabulary coverage, and dataset size estimates.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DatasetStatistics:
    """Compute and report statistics for a text dataset.

    Args:
        tokenizer: HuggingFace tokenizer instance used for length computation.
        sample_size: Maximum number of documents to analyse (default 10 000).
            Set to ``None`` to analyse all documents.

    Example:
        >>> stats = DatasetStatistics(tokenizer=tokenizer, sample_size=5000)
        >>> report = stats.compute(texts)
        >>> stats.print_report(report)
    """

    def __init__(
        self,
        tokenizer: Any,
        sample_size: Optional[int] = 10_000,
    ):
        self.tokenizer = tokenizer
        self.sample_size = sample_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, texts: List[str]) -> Dict[str, Any]:
        """Compute statistics for a list of text documents.

        Args:
            texts: List of raw text strings.

        Returns:
            Dictionary with keys: ``num_documents``, ``num_tokens``,
            ``char_lengths``, ``token_lengths``, ``vocab_coverage``.
        """
        if self.sample_size is not None:
            texts = texts[: self.sample_size]

        char_lengths: List[int] = []
        token_lengths: List[int] = []
        token_counter: Dict[int, int] = {}

        for text in texts:
            char_lengths.append(len(text))
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_lengths.append(len(ids))
            for tok in ids:
                token_counter[tok] = token_counter.get(tok, 0) + 1

        total_tokens = sum(token_lengths)
        vocab_size = self.tokenizer.vocab_size
        unique_tokens = len(token_counter)
        vocab_coverage = unique_tokens / vocab_size if vocab_size > 0 else 0.0

        tl = np.array(token_lengths, dtype=np.float64) if token_lengths else np.array([0.0])
        cl = np.array(char_lengths, dtype=np.float64) if char_lengths else np.array([0.0])

        return {
            "num_documents": len(texts),
            "num_tokens": total_tokens,
            "token_lengths": {
                "min": int(tl.min()),
                "max": int(tl.max()),
                "mean": float(tl.mean()),
                "median": float(np.median(tl)),
                "p95": float(np.percentile(tl, 95)),
                "std": float(tl.std()),
            },
            "char_lengths": {
                "min": int(cl.min()),
                "max": int(cl.max()),
                "mean": float(cl.mean()),
            },
            "vocab_coverage": vocab_coverage,
            "unique_tokens": unique_tokens,
        }

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print a human-readable summary of a statistics report."""
        print("=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        print(f"  Documents : {report['num_documents']:,}")
        print(f"  Tokens    : {report['num_tokens']:,}")
        tl = report["token_lengths"]
        print(f"  Token length  min={tl['min']}  max={tl['max']}  "
              f"mean={tl['mean']:.1f}  median={tl['median']:.1f}  "
              f"p95={tl['p95']:.1f}  std={tl['std']:.1f}")
        cl = report["char_lengths"]
        print(f"  Char length   min={cl['min']}  max={cl['max']}  mean={cl['mean']:.1f}")
        print(f"  Vocab coverage: {report['vocab_coverage']:.2%}  "
              f"({report['unique_tokens']:,} unique tokens)")
        print("=" * 50)

    def estimate_training_tokens(
        self,
        texts: List[str],
        max_seq_length: int = 2048,
    ) -> Dict[str, Any]:
        """Estimate the number of training chunks for a given sequence length.

        Args:
            texts: List of raw text strings.
            max_seq_length: Chunk size used during training.

        Returns:
            Dictionary with ``total_tokens``, ``num_chunks``, and
            ``estimated_steps_per_epoch`` (for a batch size of 1).
        """
        report = self.compute(texts)
        total_tokens = report["num_tokens"]
        num_chunks = total_tokens // max_seq_length
        return {
            "total_tokens": total_tokens,
            "num_chunks": num_chunks,
            "max_seq_length": max_seq_length,
            "estimated_steps_per_epoch": num_chunks,
        }
