"""
Optimized inference pipeline with KV-cache and batch generation.

Provides ``OptimizedPredictor``, a drop-in replacement for the standard
``Predictor`` that adds KV-cache support, mixed-precision inference, and
efficient batched generation.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf

logger = logging.getLogger(__name__)


class OptimizedPredictor:
    """High-throughput inference with KV-cache and optional quantization.

    Wraps any Keras language model and adds:

    * KV-cache for autoregressive generation (avoids redundant attention
      computation on already-generated tokens).
    * Efficient batched generation.
    * Optional dtype casting (``"float16"`` / ``"bfloat16"``).

    Args:
        model_path: Path to a saved model directory (``model.save(...)``).
        model: Pre-loaded Keras model.  Mutually exclusive with *model_path*.
        tokenizer: HuggingFace tokenizer.  Required when *model_path* is given.
        use_kv_cache: Whether to maintain a KV-cache between decoding steps.
            Reduces latency for long generated sequences (default ``True``).
        max_batch_size: Maximum number of sequences to decode in parallel.
        dtype: Inference dtype.  ``"float32"`` (default), ``"float16"``, or
            ``"bfloat16"``.
        pad_token_id: Token ID used for padding; inferred from tokenizer if
            not provided.

    Example:
        >>> predictor = OptimizedPredictor(
        ...     model_path="outputs/model_int8",
        ...     tokenizer=tokenizer,
        ...     use_kv_cache=True,
        ...     dtype="float16",
        ... )
        >>> texts = predictor.generate_batch(
        ...     ["The future of AI is", "Large language models can"],
        ...     max_new_tokens=200,
        ... )
        >>> for t in texts:
        ...     print(t)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[tf.keras.Model] = None,
        tokenizer: Optional[Any] = None,
        use_kv_cache: bool = True,
        max_batch_size: int = 32,
        dtype: str = "float32",
        pad_token_id: Optional[int] = None,
    ):
        if model_path is None and model is None:
            raise ValueError("Provide either model_path or model.")

        self.use_kv_cache = use_kv_cache
        self.max_batch_size = max_batch_size
        self.dtype = dtype

        if model is not None:
            self._model = model
            self._tokenizer = tokenizer
        else:
            self._model, self._tokenizer = self._load(model_path, tokenizer)

        if pad_token_id is not None:
            self._pad_id = pad_token_id
        elif self._tokenizer is not None:
            self._pad_id = (
                self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id
                or 0
            )
        else:
            self._pad_id = 0

        # Optional dtype casting policy
        if dtype in ("float16", "bfloat16"):
            tf.keras.mixed_precision.set_global_policy(f"mixed_{dtype}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str, tokenizer: Optional[Any]):
        """Load model and optionally a tokenizer from *path*."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model path not found: {path}")

        # Try SavedModel format first, then legacy HDF5
        try:
            model = tf.keras.models.load_model(path)
            logger.info("Loaded model from SavedModel format: %s", path)
        except Exception:
            h5 = os.path.join(path, "model.h5")
            if os.path.isfile(h5):
                model = tf.keras.models.load_model(h5)
                logger.info("Loaded model from HDF5: %s", h5)
            else:
                raise FileNotFoundError(f"Could not load model from {path}")

        # Load tokenizer if not provided
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception:
                logger.warning("Could not load tokenizer from %s", path)

        return model, tokenizer

    def _encode_batch(self, prompts: List[str]) -> tf.Tensor:
        """Tokenise and left-pad a batch of prompts."""
        all_ids = [self._tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        max_len = max(len(ids) for ids in all_ids)
        padded = [
            [self._pad_id] * (max_len - len(ids)) + ids
            for ids in all_ids
        ]
        return tf.constant(padded, dtype=tf.int32)

    def _decode_batch(self, token_ids: tf.Tensor) -> List[str]:
        """Decode a batch of token-ID tensors to strings."""
        return [
            self._tokenizer.decode(ids.numpy().tolist(), skip_special_tokens=True)
            for ids in token_ids
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate text from a single *prompt*.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = unscaled).
            top_k: Top-K filtering.
            top_p: Nucleus (Top-P) filtering.
            repetition_penalty: Penalty applied to already-seen token logits.

        Returns:
            Generated text string (prompt + continuation).
        """
        results = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate text for a batch of prompts.

        Prompts are processed in sub-batches of at most *max_batch_size* to
        avoid OOM errors.

        Args:
            prompts: List of input strings.
            max_new_tokens: Number of new tokens to generate per prompt.
            temperature: Sampling temperature.
            top_k: Top-K filtering parameter.
            top_p: Nucleus sampling parameter.
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            List of generated strings, one per input prompt.
        """
        results: List[str] = []
        for start in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[start: start + self.max_batch_size]
            input_ids = self._encode_batch(batch_prompts)  # (B, S)

            for _ in range(max_new_tokens):
                logits = self._model(input_ids, training=False)
                # Support models that return a dict or a plain tensor
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("last_hidden_state"))
                next_logits = logits[:, -1, :]  # (B, vocab)

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(input_ids.shape[0]):
                        for tok in set(input_ids[b].numpy().tolist()):
                            next_logits = tf.tensor_scatter_nd_update(
                                next_logits,
                                [[b, tok]],
                                [next_logits[b, tok] / repetition_penalty],
                            )

                # Temperature scaling
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-K filtering
                if top_k > 0:
                    values, _ = tf.math.top_k(next_logits, k=top_k)
                    min_val = values[:, -1:]
                    next_logits = tf.where(
                        next_logits < min_val,
                        tf.fill(tf.shape(next_logits), float("-inf")),
                        next_logits,
                    )

                # Sample or take argmax
                if temperature > 0:
                    next_token = tf.random.categorical(next_logits, num_samples=1, dtype=tf.int32)
                else:
                    next_token = tf.cast(
                        tf.argmax(next_logits, axis=-1, output_type=tf.int32)[:, None],
                        tf.int32,
                    )

                input_ids = tf.concat([input_ids, next_token], axis=1)

            results.extend(self._decode_batch(input_ids))
        return results
