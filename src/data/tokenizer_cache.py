"""
Disk-backed tokenization cache.

Caches encoded token-ID sequences on disk so that repeated runs over the
same text do not incur tokenization overhead.  The cache is keyed by the
SHA-256 hash of the raw text, and each entry is stored as a compressed
NumPy array inside a directory tree.
"""

import hashlib
import logging
import os
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TokenizerCache:
    """Disk-backed cache for tokenized text.

    Stores tokenized sequences as compressed NumPy arrays to avoid
    re-tokenizing the same text across multiple training runs.

    Args:
        cache_dir: Root directory for cache files.
        tokenizer: HuggingFace tokenizer instance.
        add_special_tokens: Whether to add special tokens during encoding
            (default ``True``).

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> cache = TokenizerCache(cache_dir=".token_cache", tokenizer=tokenizer)
        >>> token_ids = cache.encode("Hello world!")
        >>> token_ids
        [15496, 995, 0]
    """

    def __init__(
        self,
        cache_dir: str,
        tokenizer: Any,
        add_special_tokens: bool = True,
    ):
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        os.makedirs(cache_dir, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(text: str) -> str:
        """Return the SHA-256 hex digest of *text* (used as the cache key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        """Return the file path for a given cache key."""
        return os.path.join(self.cache_dir, key[:2], f"{key}.npz")

    def _load(self, path: str) -> Optional[List[int]]:
        """Load a cached token-ID list from *path*, or return ``None``."""
        try:
            data = np.load(path)
            return data["ids"].tolist()
        except Exception:
            return None

    def _save(self, path: str, ids: List[int]) -> None:
        """Save token-IDs to *path* as a compressed NumPy array."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, ids=np.array(ids, dtype=np.int32))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Return token IDs for *text*, using the cache when available.

        Args:
            text: Raw text string to tokenize.

        Returns:
            List of integer token IDs.
        """
        key = self._key(text)
        path = self._cache_path(key)
        cached = self._load(path)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        ids: List[int] = self.tokenizer.encode(
            text, add_special_tokens=self.add_special_tokens
        )
        self._save(path, ids)
        return ids

    def cache_stats(self) -> dict:
        """Return hit/miss statistics for the current session."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Delete all cached files under *cache_dir*."""
        import shutil

        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        self._hits = 0
        self._misses = 0
        logger.info("TokenizerCache cleared: %s", self.cache_dir)
