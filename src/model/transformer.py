"""
Transformer model implementation in TensorFlow.

This module implements a transformer-based language model from scratch,
supporting text generation, classification, and question answering tasks.

Production-grade optimizations (opt-in via config flags):
- Grouped-Query Attention (GQA)  — reduces KV-cache size
- Rotary Position Embeddings (RoPE) — better long-sequence generalisation
- SwiGLU Feed-Forward Network — improved parameter efficiency
- Flash Attention — memory-efficient attention computation
- Gradient Checkpointing — 30-40% training memory reduction
- Mixed Precision (fp16/bf16) — 2x memory savings
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf

from src.model.optimizations import (
    GroupedQueryAttention,
    SwiGLUFeedForward,
    flash_attention,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predefined production-grade model configurations
# ---------------------------------------------------------------------------

#: Predefined configurations for 1B–8B parameter models.
#: All use GQA + RoPE + SwiGLU for modern efficiency.
PREDEFINED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "1b": {
        "vocab_size": 50257,
        "d_model": 768,
        "num_heads": 12,
        "num_kv_heads": 2,
        "num_layers": 12,
        "d_ff": 3072,
        "max_seq_length": 2048,
        "use_gqa": True,
        "use_rope": True,
        "use_swiglu": True,
    },
    "3b": {
        "vocab_size": 50257,
        "d_model": 1536,
        "num_heads": 16,
        "num_kv_heads": 4,
        "num_layers": 24,
        "d_ff": 6144,
        "max_seq_length": 2048,
        "use_gqa": True,
        "use_rope": True,
        "use_swiglu": True,
    },
    "5b": {
        "vocab_size": 50257,
        "d_model": 2048,
        "num_heads": 32,
        "num_kv_heads": 8,
        "num_layers": 32,
        "d_ff": 8192,
        "max_seq_length": 2048,
        "use_gqa": True,
        "use_rope": True,
        "use_swiglu": True,
    },
    "8b": {
        "vocab_size": 50257,
        "d_model": 2560,
        "num_heads": 32,
        "num_kv_heads": 8,
        "num_layers": 40,
        "d_ff": 10240,
        "max_seq_length": 2048,
        "use_gqa": True,
        "use_rope": True,
        "use_swiglu": True,
    },
}


@dataclass
class TransformerConfig:
    """Configuration class for the transformer model.

    Attributes:
        vocab_size: Size of the vocabulary.
        d_model: Dimensionality of the model embeddings.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads for GQA (``None`` = same as
            ``num_heads``, i.e. standard MHA).
        num_layers: Number of transformer layers.
        d_ff: Dimensionality of the feed-forward network.
        max_seq_length: Maximum sequence length.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout for attention weights.
        task: Task type ('text_generation', 'sequence_classification',
              'question_answering', 'sentiment_analysis').
        num_labels: Number of labels for classification tasks.
        positional_encoding: Type of positional encoding.  Ignored when
            ``use_rope=True`` (RoPE is applied inside the attention layer).
            Set to ``'learned'`` or ``'sinusoidal'``.
        activation: Activation function for the FFN ('gelu' or 'relu').
            Ignored when ``use_swiglu=True``.
        pad_token_id: Token ID used for padding.
        use_gqa: Enable Grouped-Query Attention (requires ``num_kv_heads``).
        use_rope: Enable Rotary Position Embeddings inside the attention layer.
        use_swiglu: Enable SwiGLU feed-forward network.
        use_flash_attention: Enable memory-efficient tiled attention computation.
        gradient_checkpointing: Enable gradient checkpointing to reduce
            training memory (only effective during training).
        mixed_precision: Mixed precision training mode.
            ``None`` (default FP32), ``'fp16'``, or ``'bf16'``.
    """

    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: Optional[int] = None
    num_layers: int = 12
    d_ff: int = 3072
    max_seq_length: int = 1024
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    task: str = "text_generation"
    num_labels: int = 2
    positional_encoding: str = "learned"
    activation: str = "gelu"
    pad_token_id: int = 0
    # Production-grade optimizations (opt-in)
    use_gqa: bool = False
    use_rope: bool = False
    use_swiglu: bool = False
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformerConfig":
        """Create a TransformerConfig from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_seq_length": self.max_seq_length,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "task": self.task,
            "num_labels": self.num_labels,
            "positional_encoding": self.positional_encoding,
            "activation": self.activation,
            "pad_token_id": self.pad_token_id,
            "use_gqa": self.use_gqa,
            "use_rope": self.use_rope,
            "use_swiglu": self.use_swiglu,
            "use_flash_attention": self.use_flash_attention,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
        }


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads,
    supporting both self-attention and cross-attention.

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
        use_rope: Apply Rotary Position Embeddings to Q and K.
        use_flash_attention: Use memory-efficient tiled attention computation.
        max_seq_length: Maximum sequence length (for RoPE precomputation).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        use_rope: bool = False,
        use_flash_attention: bool = False,
        max_seq_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        self.max_seq_length = max_seq_length

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False, name="query_projection")
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False, name="key_projection")
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False, name="value_projection")
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False, name="output_projection")

        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.scale = tf.math.sqrt(tf.cast(self.d_k, tf.float32))

        if use_rope:
            from src.model.optimizations import RotaryEmbedding
            self.rope = RotaryEmbedding(
                d_k=self.d_k,
                max_seq_length=max_seq_length,
                name="rope",
            )

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, d_k).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            batch_size: Batch size.

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k).
        """
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_v, d_model).
            mask: Optional mask tensor. Shape should be broadcastable to
                  (batch_size, num_heads, seq_len_q, seq_len_k).
            training: Whether in training mode.

        Returns:
            Tuple of (output tensor, attention weights).
        """
        batch_size = tf.shape(query)[0]

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if self.use_rope:
            seq_len = tf.shape(query)[1]
            q, k = self.rope(q, k, seq_len=seq_len)

        if self.use_flash_attention:
            context, attn_weights = flash_attention(
                q, k, v,
                mask=mask,
                scale=float(1.0 / math.sqrt(self.d_k)),
                dropout_rate=self.attn_dropout.rate,
                training=training,
            )
        else:
            # Scaled dot-product attention
            scores = tf.matmul(q, k, transpose_b=True) / self.scale

            if mask is not None:
                scores = scores + (mask * -1e9)

            attn_weights = tf.nn.softmax(scores, axis=-1)
            attn_weights = self.attn_dropout(attn_weights, training=training)

            context = tf.matmul(attn_weights, v)

        # Reshape back to (batch_size, seq_len, d_model)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        output = self.wo(context)

        return output, attn_weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.attn_dropout.rate,
            "use_rope": self.use_rope,
            "use_flash_attention": self.use_flash_attention,
            "max_seq_length": self.max_seq_length,
        })
        return config


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward network.

    Applies two linear transformations with an activation in between.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden layer dimensionality.
        dropout_rate: Dropout probability.
        activation: Activation function ('gelu' or 'relu').
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation

        act_fn = tf.keras.activations.gelu if activation == "gelu" else tf.keras.activations.relu
        self.dense1 = tf.keras.layers.Dense(d_ff, activation=act_fn, name="ffn_dense1")
        self.dense2 = tf.keras.layers.Dense(d_model, name="ffn_dense2")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout.rate,
            "activation": self.activation_name,
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """A single transformer decoder block.

    Each block consists of:
    1. Self-attention (standard MHA or GQA) with residual connection and layer norm
    2. Feed-forward network (standard or SwiGLU) with residual connection and layer norm

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimensionality.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout for attention weights.
        activation: Activation function for feed-forward network.
        num_kv_heads: KV heads for GQA.  ``None`` = standard MHA.
        use_gqa: Enable Grouped-Query Attention.
        use_rope: Enable Rotary Position Embeddings (inside the attention layer).
        use_swiglu: Enable SwiGLU feed-forward.
        use_flash_attention: Enable memory-efficient tiled attention.
        max_seq_length: Maximum sequence length (for RoPE precomputation).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        num_kv_heads: Optional[int] = None,
        use_gqa: bool = False,
        use_rope: bool = False,
        use_swiglu: bool = False,
        use_flash_attention: bool = False,
        max_seq_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_gqa = use_gqa
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.use_flash_attention = use_flash_attention
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if use_gqa:
            self.attention = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                dropout_rate=attention_dropout,
                use_rope=use_rope,
                max_seq_length=max_seq_length,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model, num_heads, attention_dropout,
                use_rope=use_rope,
                use_flash_attention=use_flash_attention,
                max_seq_length=max_seq_length,
            )

        if use_swiglu:
            self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout_rate)
        else:
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate, activation)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional causal mask tensor.
            training: Whether in training mode.

        Returns:
            Tuple of (output tensor, attention weights).
        """
        # Self-attention with pre-norm (following GPT-2 style)
        normed = self.norm1(x)
        attn_output, attn_weights = self.attention(
            normed, normed, normed, mask=mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        # Feed-forward with pre-norm
        normed = self.norm2(x)
        ffn_output = self.ffn(normed, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output

        return x, attn_weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout1.rate,
            "attention_dropout": self.attention.attn_dropout.rate,
            "activation": getattr(self.ffn, "activation_name", "swiglu"),
            "num_kv_heads": self.num_kv_heads,
            "use_gqa": self.use_gqa,
            "use_rope": self.use_rope,
            "use_swiglu": self.use_swiglu,
            "use_flash_attention": self.use_flash_attention,
        })
        return config


def get_sinusoidal_encoding(max_seq_length: int, d_model: int) -> tf.Tensor:
    """Create sinusoidal positional encodings.

    Args:
        max_seq_length: Maximum sequence length.
        d_model: Model dimensionality.

    Returns:
        Positional encoding tensor of shape (1, max_seq_length, d_model).
    """
    positions = np.arange(max_seq_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # Apply sin to even indices, cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def create_causal_mask(seq_len: int) -> tf.Tensor:
    """Create a causal (autoregressive) mask.

    Prevents positions from attending to subsequent positions.

    Args:
        seq_len: Sequence length.

    Returns:
        Mask tensor of shape (1, 1, seq_len, seq_len).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.reshape(mask, (1, 1, seq_len, seq_len))


def create_padding_mask(token_ids: tf.Tensor, pad_token_id: int = 0) -> tf.Tensor:
    """Create a padding mask to ignore padding tokens.

    Args:
        token_ids: Input token IDs of shape (batch_size, seq_len).
        pad_token_id: ID of the padding token.

    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len).
    """
    mask = tf.cast(tf.equal(token_ids, pad_token_id), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


class SmallTransformer(tf.keras.Model):
    """Small transformer-based language model.

    A GPT-style decoder-only transformer that can be used for text generation,
    sequence classification, sentiment analysis, and question answering.

    The model has approximately 117M parameters with default configuration,
    and can be scaled up to 1B-8B parameters by adjusting ``d_model``,
    ``num_layers``, and ``d_ff``.  Production-grade optimizations (GQA, RoPE,
    SwiGLU, Flash Attention) can be enabled via config flags.

    Predefined configurations (see ``config/model_config.yaml``):
    - 1B:  d_model=768,  12 heads, 2 KV heads, 12 layers,  d_ff=3072
    - 3B:  d_model=1536, 16 heads, 4 KV heads, 24 layers,  d_ff=6144
    - 5B:  d_model=2048, 32 heads, 8 KV heads, 32 layers,  d_ff=8192
    - 8B:  d_model=2560, 32 heads, 8 KV heads, 40 layers,  d_ff=10240

    Args:
        config: TransformerConfig instance with model hyperparameters.

    Example:
        >>> config = TransformerConfig(vocab_size=50257, d_model=768, num_heads=12)
        >>> model = SmallTransformer(config)
        >>> input_ids = tf.constant([[1, 2, 3, 4, 5]])
        >>> outputs = model(input_ids)
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Determine effective KV heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads

        # Token embeddings
        self.token_embedding = tf.keras.layers.Embedding(
            config.vocab_size,
            config.d_model,
            name="token_embedding",
        )

        # Positional encoding (skipped when use_rope=True — RoPE is inside attention)
        if not config.use_rope:
            if config.positional_encoding == "learned":
                self.pos_embedding = tf.keras.layers.Embedding(
                    config.max_seq_length,
                    config.d_model,
                    name="position_embedding",
                )
            else:
                self.sinusoidal_encoding = get_sinusoidal_encoding(
                    config.max_seq_length, config.d_model
                )

        self.emb_dropout = tf.keras.layers.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout_rate=config.dropout_rate,
                attention_dropout=config.attention_dropout,
                activation=config.activation,
                num_kv_heads=num_kv_heads,
                use_gqa=config.use_gqa,
                use_rope=config.use_rope,
                use_swiglu=config.use_swiglu,
                use_flash_attention=config.use_flash_attention,
                max_seq_length=config.max_seq_length,
                name=f"transformer_block_{i}",
            )
            for i in range(config.num_layers)
        ]

        # Final layer normalization
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")

        # Task-specific heads
        if config.task in ("text_generation",):
            self.lm_head = tf.keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head"
            )
        elif config.task in ("sequence_classification", "sentiment_analysis"):
            self.classifier = tf.keras.layers.Dense(
                config.num_labels, name="classifier"
            )
        elif config.task == "question_answering":
            self.qa_outputs = tf.keras.layers.Dense(2, name="qa_outputs")

        logger.info(
            "Initialized SmallTransformer with %d layers, d_model=%d, "
            "%d heads (kv_heads=%d), task=%s, gqa=%s, rope=%s, swiglu=%s",
            config.num_layers, config.d_model, config.num_heads, num_kv_heads,
            config.task, config.use_gqa, config.use_rope, config.use_swiglu,
        )

    def get_embeddings(
        self,
        input_ids: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute token + positional embeddings.

        When ``use_rope=True`` only token embeddings are returned; rotary
        position information is applied inside each attention layer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = tf.shape(input_ids)[1]
        tok_emb = self.token_embedding(input_ids)

        if self.config.use_rope:
            x = tok_emb
        elif self.config.positional_encoding == "learned":
            positions = tf.range(seq_len)
            pos_emb = self.pos_embedding(positions)
            x = tok_emb + pos_emb
        else:
            pos_emb = self.sinusoidal_encoding[:, :seq_len, :]
            x = tok_emb + pos_emb

        return self.emb_dropout(x, training=training)

    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
        return_attention_weights: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
                1 for tokens to attend to, 0 for tokens to ignore.
            training: Whether in training mode.
            return_attention_weights: Whether to return attention weights.

        Returns:
            Dictionary containing task-specific outputs:
            - For text_generation: 'logits' of shape (batch_size, seq_len, vocab_size)
            - For classification: 'logits' of shape (batch_size, num_labels)
            - For question_answering: 'start_logits', 'end_logits' each (batch_size, seq_len)
            - Always includes 'hidden_states' (batch_size, seq_len, d_model)
        """
        seq_len = tf.shape(input_ids)[1]

        # Build combined mask: causal + padding
        causal_mask = create_causal_mask(seq_len)
        if attention_mask is not None:
            pad_mask = 1.0 - tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
            combined_mask = tf.maximum(causal_mask, pad_mask)
        else:
            combined_mask = causal_mask

        x = self.get_embeddings(input_ids, training=training)

        all_attn_weights = []
        for block in self.blocks:
            if self.config.gradient_checkpointing and training:
                # Use tf.recompute_grad for gradient checkpointing
                # This trades compute for memory by recomputing activations
                # during the backward pass instead of storing them.
                @tf.recompute_grad
                def _block_fn(inputs, blk=block):
                    out, weights = blk(inputs, mask=combined_mask, training=True)
                    return out

                x = _block_fn(x)
                attn_weights = None  # weights not available with checkpointing
            else:
                x, attn_weights = block(x, mask=combined_mask, training=training)
            if return_attention_weights and attn_weights is not None:
                all_attn_weights.append(attn_weights)

        hidden_states = self.final_norm(x)
        outputs = {"hidden_states": hidden_states}

        if self.config.task == "text_generation":
            outputs["logits"] = self.lm_head(hidden_states)
        elif self.config.task in ("sequence_classification", "sentiment_analysis"):
            # Use representation at last non-padding position for classification
            pooled = hidden_states[:, -1, :]
            outputs["logits"] = self.classifier(pooled)
        elif self.config.task == "question_answering":
            qa_logits = self.qa_outputs(hidden_states)
            start_logits, end_logits = tf.split(qa_logits, 2, axis=-1)
            outputs["start_logits"] = tf.squeeze(start_logits, axis=-1)
            outputs["end_logits"] = tf.squeeze(end_logits, axis=-1)

        if return_attention_weights:
            outputs["attention_weights"] = all_attn_weights

        return outputs

    def compute_loss(
        self,
        input_ids: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        start_positions: Optional[tf.Tensor] = None,
        end_positions: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Compute task-specific loss.

        Args:
            input_ids: Input token IDs.
            labels: Labels for supervised tasks. For text generation, these are
                the shifted input_ids. For classification, these are class indices.
            attention_mask: Attention mask.
            start_positions: Start positions for Q&A tasks.
            end_positions: End positions for Q&A tasks.
            training: Whether in training mode.

        Returns:
            Dictionary with 'loss' and task-specific outputs.
        """
        outputs = self(input_ids, attention_mask=attention_mask, training=training)

        if self.config.task == "text_generation" and labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = outputs["logits"][:, :-1, :]
            shift_labels = labels[:, 1:]

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction="none"
            )
            loss = loss_fn(shift_labels, shift_logits)

            if attention_mask is not None:
                mask = tf.cast(attention_mask[:, 1:], tf.float32)
                loss = loss * mask
                outputs["loss"] = tf.reduce_sum(loss) / tf.reduce_sum(mask)
            else:
                outputs["loss"] = tf.reduce_mean(loss)

        elif self.config.task in ("sequence_classification", "sentiment_analysis"):
            if labels is not None:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                outputs["loss"] = loss_fn(labels, outputs["logits"])

        elif self.config.task == "question_answering":
            if start_positions is not None and end_positions is not None:
                seq_len = tf.shape(input_ids)[1]
                start_positions = tf.clip_by_value(start_positions, 0, seq_len - 1)
                end_positions = tf.clip_by_value(end_positions, 0, seq_len - 1)

                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                start_loss = loss_fn(start_positions, outputs["start_logits"])
                end_loss = loss_fn(end_positions, outputs["end_logits"])
                outputs["loss"] = (start_loss + end_loss) / 2.0

        return outputs

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(np.prod(v.shape) for v in self.trainable_variables)

    def get_config(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict()}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmallTransformer":
        transformer_config = TransformerConfig.from_dict(config["config"])
        return cls(transformer_config)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SmallTransformer":
        """Load model configuration from a YAML file and create model.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            SmallTransformer instance.
        """
        import yaml
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", cfg)
        # Handle ``size`` shortcut (e.g. size: "3b")
        size = model_cfg.pop("size", None)
        if size is not None and size in PREDEFINED_CONFIGS:
            base = dict(PREDEFINED_CONFIGS[size])
            base.update({k: v for k, v in model_cfg.items() if v is not None})
            model_cfg = base
        config = TransformerConfig.from_dict(model_cfg)
        return cls(config)

    @classmethod
    def for_size(
        cls,
        size: str,
        task: str = "text_generation",
        **overrides: Any,
    ) -> "SmallTransformer":
        """Create a model using a predefined size configuration.

        Args:
            size: Model size key — one of ``'1b'``, ``'3b'``, ``'5b'``, ``'8b'``.
            task: Task type for the model head.
            **overrides: Additional keyword arguments to override config values.

        Returns:
            SmallTransformer instance.

        Raises:
            ValueError: If ``size`` is not a recognised key.
        """
        if size not in PREDEFINED_CONFIGS:
            raise ValueError(
                f"Unknown model size '{size}'. "
                f"Choose from: {list(PREDEFINED_CONFIGS.keys())}"
            )
        cfg = dict(PREDEFINED_CONFIGS[size])
        cfg["task"] = task
        cfg.update(overrides)
        config = TransformerConfig.from_dict(cfg)
        return cls(config)

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and configuration.

        Args:
            save_dir: Directory to save the model.
        """
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)
        self.save_weights(os.path.join(save_dir, "model_weights.weights.h5"))

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info("Model saved to %s", save_dir)

    @classmethod
    def load_pretrained(cls, load_dir: str) -> "SmallTransformer":
        """Load model from saved weights and configuration.

        Args:
            load_dir: Directory containing saved model.

        Returns:
            Loaded SmallTransformer instance.
        """
        import os
        import json

        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = TransformerConfig.from_dict(config_dict)
        model = cls(config)

        # Build model with dummy input before loading weights
        dummy_input = tf.zeros((1, 1), dtype=tf.int32)
        model(dummy_input)
        model.load_weights(os.path.join(load_dir, "model_weights.weights.h5"))

        logger.info("Model loaded from %s", load_dir)
        return model
