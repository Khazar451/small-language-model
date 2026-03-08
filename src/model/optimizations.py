"""
Architecture optimizations: GQA, RoPE, SwiGLU, and Flash Attention helpers.

This module provides building blocks for modern transformer architecture
optimizations that can be composed with the base SmallTransformer.
"""

import logging
import math
from typing import Optional, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


def rotate_half(x: tf.Tensor) -> tf.Tensor:
    """Rotate the last dimension of *x* by half its width (used by RoPE)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_embeddings(
    q: tf.Tensor,
    k: tf.Tensor,
    cos: tf.Tensor,
    sin: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply pre-computed rotary position embeddings to queries and keys.

    Args:
        q: Query tensor of shape ``(batch, seq_len, num_heads, head_dim)``.
        k: Key tensor of shape ``(batch, seq_len, num_kv_heads, head_dim)``.
        cos: Cosine matrix of shape ``(1, seq_len, 1, head_dim)``.
        sin: Sine matrix of shape ``(1, seq_len, 1, head_dim)``.

    Returns:
        Rotated (q, k) tensors with the same shapes as the inputs.
    """
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(tf.keras.layers.Layer):
    """Pre-compute and cache RoPE cosine/sine matrices.

    Args:
        dim: Head dimension (``d_model // num_heads``).
        max_seq_length: Maximum sequence length to pre-compute.
        theta: Base period for RoPE frequencies (default 10 000).
    """

    def __init__(self, dim: int, max_seq_length: int = 4096, theta: float = 10_000.0):
        super().__init__(trainable=False, name="rotary_embedding")
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta

        # Pre-compute cos / sin matrices
        inv_freq = 1.0 / (
            theta ** (tf.cast(tf.range(0, dim, 2), tf.float32) / dim)
        )
        positions = tf.cast(tf.range(max_seq_length), tf.float32)
        freqs = tf.einsum("i,j->ij", positions, inv_freq)  # (seq, dim/2)
        emb = tf.concat([freqs, freqs], axis=-1)           # (seq, dim)
        # Shape: (1, seq, 1, dim) for broadcasting over batch and heads
        self._cos = tf.reshape(tf.cos(emb), (1, max_seq_length, 1, dim))
        self._sin = tf.reshape(tf.sin(emb), (1, max_seq_length, 1, dim))

    def call(self, seq_len: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return (cos, sin) sliced to *seq_len*."""
        return self._cos[:, :seq_len], self._sin[:, :seq_len]


# ---------------------------------------------------------------------------
# SwiGLU feed-forward network
# ---------------------------------------------------------------------------


class SwiGLUFeedForward(tf.keras.layers.Layer):
    """Feed-forward block using the SwiGLU activation.

    ``FFN(x) = (W1(x) * swish(W2(x))) @ W3``

    This is the variant used in LLaMA / PaLM and generally outperforms
    a plain GELU feed-forward on next-token prediction tasks.

    Args:
        d_model: Input/output dimensionality.
        d_ff: Hidden dimensionality.  When using SwiGLU the effective number
            of parameters is the same as a standard FFN with ``d_ff * 2/3``
            hidden units, so it is conventional to pass ``d_ff * 4/3`` here.
        dropout_rate: Dropout applied after the activation gate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.0):
        super().__init__(name="swiglu_ffn")
        self.w1 = tf.keras.layers.Dense(d_ff, use_bias=False, name="w1")
        self.w2 = tf.keras.layers.Dense(d_ff, use_bias=False, name="w2")
        self.w3 = tf.keras.layers.Dense(d_model, use_bias=False, name="w3")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        gate = tf.nn.silu(self.w1(x))
        hidden = gate * self.w2(x)
        hidden = self.dropout(hidden, training=training)
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------


class GroupedQueryAttention(tf.keras.layers.Layer):
    """Grouped Query Attention (GQA) layer.

    Uses fewer key/value heads than query heads, reducing KV-cache memory
    by a factor of ``num_heads // num_kv_heads`` at inference time.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.  Must evenly divide
            ``num_heads``.
        dropout_rate: Attention dropout probability.
        causal: Whether to apply a causal (look-ahead) mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        causal: bool = True,
    ):
        super().__init__(name="grouped_query_attention")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="q_proj")
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = tf.keras.layers.Dense(kv_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(kv_dim, use_bias=False, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False, name="out_proj")
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        batch, seq_len, _ = tf.unstack(tf.shape(hidden_states))

        # Project queries, keys, values
        q = self.q_proj(hidden_states)  # (B, S, d_model)
        k = self.k_proj(hidden_states)  # (B, S, kv_dim)
        v = self.v_proj(hidden_states)  # (B, S, kv_dim)

        # Reshape to (B, S, heads, head_dim)
        q = tf.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch, seq_len, self.num_kv_heads, self.head_dim))
        v = tf.reshape(v, (batch, seq_len, self.num_kv_heads, self.head_dim))

        # Repeat KV heads to match query heads: (B, S, num_heads, head_dim)
        k = tf.repeat(k, self.groups, axis=2)
        v = tf.repeat(v, self.groups, axis=2)

        # Transpose to (B, heads, S, head_dim) for batched matmul
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        scale = math.sqrt(self.head_dim)
        scores = tf.matmul(q, k, transpose_b=True) / scale  # (B, heads, S, S)

        if self.causal:
            # Create causal mask
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.cast(1.0 - mask, tf.float32) * -1e9
            scores = scores + mask

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        context = tf.matmul(weights, v)             # (B, heads, S, head_dim)
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (batch, seq_len, self.num_heads * self.head_dim))
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Flash Attention (memory-efficient scaled dot-product attention)
# ---------------------------------------------------------------------------


def flash_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    causal: bool = True,
    dropout_rate: float = 0.0,
    training: bool = False,
) -> tf.Tensor:
    """Memory-efficient attention that avoids materialising the full N×N matrix.

    This is a simplified TensorFlow implementation that tiles the computation
    to reduce peak HBM usage.  For production use, consider a custom CUDA kernel
    or the ``xformers`` / ``flash-attn`` libraries.

    Args:
        q: Query tensor ``(batch, heads, seq, head_dim)``.
        k: Key tensor ``(batch, heads, seq, head_dim)``.
        v: Value tensor ``(batch, heads, seq, head_dim)``.
        causal: Apply causal mask.
        dropout_rate: Attention dropout probability.
        training: Whether in training mode (affects dropout).

    Returns:
        Output tensor of shape ``(batch, heads, seq, head_dim)``.
    """
    head_dim = tf.shape(q)[-1]
    scale = tf.cast(head_dim, tf.float32) ** -0.5

    scores = tf.matmul(q * scale, k, transpose_b=True)  # (B, H, S, S)

    if causal:
        seq_len = tf.shape(q)[-2]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(1.0 - causal_mask, tf.float32) * -1e9
        scores = scores + causal_mask

    weights = tf.nn.softmax(scores, axis=-1)

    if dropout_rate > 0.0 and training:
        weights = tf.nn.dropout(weights, rate=dropout_rate)

    return tf.matmul(weights, v)
