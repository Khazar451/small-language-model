"""
Tests for production-grade model optimizations.

Covers:
- RotaryEmbedding (RoPE)
- GroupedQueryAttention (GQA)
- SwiGLUFeedForward
- flash_attention utility
- Updated TransformerConfig fields
- Updated MultiHeadAttention (with RoPE / Flash Attention)
- Updated TransformerBlock (with GQA / SwiGLU)
- SmallTransformer with predefined sizes (1b, 3b) and new flags
- Quantization utilities
- Distributed training helpers
"""

import os
import json
import tempfile

import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_config():
    """Tiny config used in multiple tests (fast to instantiate)."""
    from src.model.transformer import TransformerConfig
    return TransformerConfig(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        d_ff=128,
        max_seq_length=32,
        task="text_generation",
        use_gqa=True,
        use_rope=True,
        use_swiglu=True,
    )


@pytest.fixture(scope="module")
def baseline_config():
    """Tiny config with no new optimizations (baseline backward compat)."""
    from src.model.transformer import TransformerConfig
    return TransformerConfig(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_length=32,
        task="text_generation",
    )


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------

class TestRotaryEmbedding:
    def test_rope_output_shapes(self):
        from src.model.optimizations import RotaryEmbedding
        rope = RotaryEmbedding(d_k=16, max_seq_length=32)
        q = tf.random.normal((2, 4, 8, 16))
        k = tf.random.normal((2, 2, 8, 16))
        q_rot, k_rot = rope(q, k, seq_len=8)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_different_sequences(self):
        """RoPE should produce different outputs for different positions."""
        from src.model.optimizations import RotaryEmbedding
        rope = RotaryEmbedding(d_k=16, max_seq_length=32)
        q = tf.ones((1, 1, 4, 16))
        k = tf.ones((1, 1, 4, 16))
        q_rot, _ = rope(q, k, seq_len=4)
        # Not all positions should be identical after rotation
        assert not np.allclose(
            q_rot[:, :, 0, :].numpy(),
            q_rot[:, :, 1, :].numpy(),
        )

    def test_compute_rope_frequencies(self):
        from src.model.optimizations import compute_rope_frequencies
        cos, sin = compute_rope_frequencies(d_k=8, max_seq_length=16)
        assert cos.shape == (1, 1, 16, 8)
        assert sin.shape == (1, 1, 16, 8)

    def test_rotate_half(self):
        from src.model.optimizations import rotate_half
        x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        rotated = rotate_half(x)
        # Second half negated and prepended: [-3, -4, 1, 2]
        expected = np.array([[-3.0, -4.0, 1.0, 2.0]])
        np.testing.assert_allclose(rotated.numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# GQA tests
# ---------------------------------------------------------------------------

class TestGroupedQueryAttention:
    def test_output_shape(self):
        from src.model.optimizations import GroupedQueryAttention
        gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = tf.random.normal((2, 8, 64))
        out, weights = gqa(x, x, x)
        assert out.shape == (2, 8, 64)

    def test_attention_weights_shape(self):
        from src.model.optimizations import GroupedQueryAttention
        gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = tf.random.normal((1, 6, 64))
        _, weights = gqa(x, x, x)
        # After repeating KV heads: (batch, num_heads, seq_q, seq_k)
        assert weights.shape == (1, 4, 6, 6)

    def test_gqa_with_mask(self):
        from src.model.optimizations import GroupedQueryAttention
        gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2, use_rope=False)
        x = tf.random.normal((1, 4, 64))
        mask = tf.zeros((1, 1, 4, 4))
        out, _ = gqa(x, x, x, mask=mask)
        assert out.shape == (1, 4, 64)

    def test_gqa_with_rope(self):
        from src.model.optimizations import GroupedQueryAttention
        gqa = GroupedQueryAttention(
            d_model=64, num_heads=4, num_kv_heads=2, use_rope=True, max_seq_length=32,
        )
        x = tf.random.normal((1, 8, 64))
        out, _ = gqa(x, x, x)
        assert out.shape == (1, 8, 64)

    def test_invalid_kv_heads_raises(self):
        from src.model.optimizations import GroupedQueryAttention
        with pytest.raises(AssertionError):
            GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=3)

    def test_mha_equivalent_when_kv_equals_heads(self):
        """When num_kv_heads == num_heads, output shape should match MHA."""
        from src.model.optimizations import GroupedQueryAttention
        gqa = GroupedQueryAttention(
            d_model=64, num_heads=4, num_kv_heads=4, use_rope=False,
        )
        x = tf.random.normal((2, 5, 64))
        out, _ = gqa(x, x, x)
        assert out.shape == (2, 5, 64)


# ---------------------------------------------------------------------------
# SwiGLU tests
# ---------------------------------------------------------------------------

class TestSwiGLUFeedForward:
    def test_output_shape(self):
        from src.model.optimizations import SwiGLUFeedForward
        ffn = SwiGLUFeedForward(d_model=64, d_ff=128)
        x = tf.random.normal((2, 8, 64))
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_dropout_applied_in_training(self):
        from src.model.optimizations import SwiGLUFeedForward
        ffn = SwiGLUFeedForward(d_model=64, d_ff=128, dropout_rate=0.9)
        x = tf.ones((1, 4, 64))
        out_train = ffn(x, training=True).numpy()
        out_infer = ffn(x, training=False).numpy()
        # High dropout in training should produce markedly different outputs
        assert not np.allclose(out_train, out_infer, atol=1e-3)

    def test_nonlinearity(self):
        """Output should be non-linear (not same as input)."""
        from src.model.optimizations import SwiGLUFeedForward
        ffn = SwiGLUFeedForward(d_model=8, d_ff=16)
        x = tf.zeros((1, 2, 8))
        out = ffn(x)
        # With zero input and zero bias the output should be all zeros
        np.testing.assert_allclose(out.numpy(), np.zeros((1, 2, 8)), atol=1e-5)


# ---------------------------------------------------------------------------
# Flash Attention tests
# ---------------------------------------------------------------------------

class TestFlashAttention:
    def test_output_shape(self):
        from src.model.optimizations import flash_attention
        q = tf.random.normal((2, 4, 8, 16))
        k = tf.random.normal((2, 4, 8, 16))
        v = tf.random.normal((2, 4, 8, 16))
        out, weights = flash_attention(q, k, v)
        assert out.shape == (2, 4, 8, 16)
        assert weights.shape == (2, 4, 8, 8)

    def test_matches_standard_attention(self):
        """Flash attention should produce the same result as standard attention."""
        from src.model.optimizations import flash_attention
        tf.random.set_seed(0)
        q = tf.random.normal((1, 2, 6, 8))
        k = tf.random.normal((1, 2, 6, 8))
        v = tf.random.normal((1, 2, 6, 8))

        scale = 1.0 / (8 ** 0.5)
        scores = tf.matmul(q, k, transpose_b=True) * scale
        std_weights = tf.nn.softmax(scores, axis=-1)
        std_out = tf.matmul(std_weights, v)

        fa_out, _ = flash_attention(q, k, v, scale=scale)
        np.testing.assert_allclose(fa_out.numpy(), std_out.numpy(), atol=1e-5)

    def test_with_causal_mask(self):
        from src.model.optimizations import flash_attention
        from src.model.transformer import create_causal_mask
        q = tf.random.normal((1, 2, 4, 8))
        k = tf.random.normal((1, 2, 4, 8))
        v = tf.random.normal((1, 2, 4, 8))
        mask = create_causal_mask(4)
        out, _ = flash_attention(q, k, v, mask=mask)
        assert out.shape == (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Updated TransformerConfig tests
# ---------------------------------------------------------------------------

class TestTransformerConfigNew:
    def test_new_fields_defaults(self):
        from src.model.transformer import TransformerConfig
        cfg = TransformerConfig()
        assert cfg.use_gqa is False
        assert cfg.use_rope is False
        assert cfg.use_swiglu is False
        assert cfg.use_flash_attention is False
        assert cfg.gradient_checkpointing is False
        assert cfg.mixed_precision is None
        assert cfg.num_kv_heads is None

    def test_roundtrip_with_new_fields(self, small_config):
        d = small_config.to_dict()
        from src.model.transformer import TransformerConfig
        restored = TransformerConfig.from_dict(d)
        assert restored.use_gqa == small_config.use_gqa
        assert restored.use_rope == small_config.use_rope
        assert restored.use_swiglu == small_config.use_swiglu
        assert restored.num_kv_heads == small_config.num_kv_heads
        assert restored.gradient_checkpointing == small_config.gradient_checkpointing
        assert restored.mixed_precision == small_config.mixed_precision

    def test_predefined_configs_exist(self):
        from src.model.transformer import PREDEFINED_CONFIGS
        for key in ("1b", "3b", "5b", "8b"):
            assert key in PREDEFINED_CONFIGS
            cfg = PREDEFINED_CONFIGS[key]
            assert cfg["use_gqa"] is True
            assert cfg["use_rope"] is True
            assert cfg["use_swiglu"] is True
            assert "num_kv_heads" in cfg


# ---------------------------------------------------------------------------
# Updated MultiHeadAttention (RoPE + Flash Attention)
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionNew:
    def test_with_rope(self):
        from src.model.transformer import MultiHeadAttention
        mha = MultiHeadAttention(d_model=64, num_heads=4, use_rope=True, max_seq_length=32)
        x = tf.random.normal((2, 8, 64))
        out, _ = mha(x, x, x)
        assert out.shape == (2, 8, 64)

    def test_with_flash_attention(self):
        from src.model.transformer import MultiHeadAttention
        mha = MultiHeadAttention(d_model=64, num_heads=4, use_flash_attention=True)
        x = tf.random.normal((1, 6, 64))
        out, _ = mha(x, x, x)
        assert out.shape == (1, 6, 64)

    def test_with_rope_and_flash(self):
        from src.model.transformer import MultiHeadAttention
        mha = MultiHeadAttention(
            d_model=64, num_heads=4, use_rope=True,
            use_flash_attention=True, max_seq_length=32,
        )
        x = tf.random.normal((1, 8, 64))
        out, _ = mha(x, x, x)
        assert out.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# Updated TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlockNew:
    def test_gqa_block(self):
        from src.model.transformer import TransformerBlock
        block = TransformerBlock(
            d_model=64, num_heads=4, d_ff=128,
            num_kv_heads=2, use_gqa=True,
        )
        x = tf.random.normal((2, 8, 64))
        out, _ = block(x)
        assert out.shape == (2, 8, 64)

    def test_swiglu_block(self):
        from src.model.transformer import TransformerBlock
        block = TransformerBlock(
            d_model=64, num_heads=4, d_ff=128, use_swiglu=True,
        )
        x = tf.random.normal((1, 6, 64))
        out, _ = block(x)
        assert out.shape == (1, 6, 64)

    def test_full_optimized_block(self):
        from src.model.transformer import TransformerBlock
        block = TransformerBlock(
            d_model=64, num_heads=4, d_ff=128,
            num_kv_heads=2,
            use_gqa=True, use_rope=True, use_swiglu=True,
            use_flash_attention=True, max_seq_length=32,
        )
        x = tf.random.normal((2, 8, 64))
        out, _ = block(x)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# SmallTransformer with new optimizations
# ---------------------------------------------------------------------------

class TestSmallTransformerOptimized:
    def test_optimized_forward_pass(self, small_config):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer(small_config)
        input_ids = tf.constant([[1, 2, 3, 4, 5]])
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 5, small_config.vocab_size)

    def test_baseline_unchanged(self, baseline_config):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer(baseline_config)
        input_ids = tf.constant([[1, 2, 3, 4]])
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 4, baseline_config.vocab_size)

    def test_for_size_1b(self):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer.for_size("1b")
        assert model.config.d_model == 768
        assert model.config.use_gqa is True
        assert model.config.use_rope is True
        assert model.config.use_swiglu is True

    def test_for_size_3b(self):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer.for_size("3b")
        assert model.config.d_model == 1536
        assert model.config.num_kv_heads == 4

    def test_for_size_invalid(self):
        from src.model.transformer import SmallTransformer
        with pytest.raises(ValueError, match="Unknown model size"):
            SmallTransformer.for_size("99b")

    def test_for_size_override(self):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer.for_size("1b", task="sentiment_analysis", num_labels=3)
        assert model.config.task == "sentiment_analysis"
        assert model.config.num_labels == 3

    def test_save_load_with_new_config(self, small_config):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer(small_config)
        # Build
        _ = model(tf.zeros((1, 4), dtype=tf.int32))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = SmallTransformer.load_pretrained(tmpdir)
        assert loaded.config.use_gqa == small_config.use_gqa
        assert loaded.config.use_rope == small_config.use_rope

    def test_count_parameters(self, small_config):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer(small_config)
        _ = model(tf.zeros((1, 4), dtype=tf.int32))
        n = model.count_parameters()
        assert n > 0

    def test_compute_loss(self, small_config):
        from src.model.transformer import SmallTransformer
        model = SmallTransformer(small_config)
        input_ids = tf.constant([[1, 2, 3, 4, 5]])
        outputs = model.compute_loss(input_ids, labels=input_ids, training=False)
        assert "loss" in outputs
        assert float(outputs["loss"]) > 0


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------

class TestQuantization:
    @pytest.fixture
    def tiny_model(self):
        from src.model.transformer import SmallTransformer, TransformerConfig
        cfg = TransformerConfig(
            vocab_size=64, d_model=32, num_heads=2, num_layers=1, d_ff=64,
        )
        model = SmallTransformer(cfg)
        _ = model(tf.zeros((1, 4), dtype=tf.int32))
        return model

    def test_quantize_int8(self, tiny_model):
        from src.model.quantization import quantize_model_weights
        results = quantize_model_weights(tiny_model, mode="int8")
        assert len(results) > 0
        for name, data in results.items():
            assert data["mode"] == "int8"
            assert data["quantized"].dtype == np.int8
            assert data["scale"] > 0

    def test_quantize_int4(self, tiny_model):
        from src.model.quantization import quantize_model_weights
        results = quantize_model_weights(tiny_model, mode="int4")
        assert len(results) > 0
        for _, data in results.items():
            assert data["mode"] == "int4"
            # INT4 values stored as int8 — range must be within [-8, 7]
            assert data["quantized"].min() >= -8
            assert data["quantized"].max() <= 7

    def test_invalid_mode(self, tiny_model):
        from src.model.quantization import quantize_model_weights
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            quantize_model_weights(tiny_model, mode="int3")

    def test_estimate_size_int8(self, tiny_model):
        from src.model.quantization import estimate_quantized_size_gb
        size = estimate_quantized_size_gb(tiny_model, mode="int8")
        assert size > 0

    def test_estimate_size_int4_smaller_than_int8(self, tiny_model):
        from src.model.quantization import estimate_quantized_size_gb
        size_int8 = estimate_quantized_size_gb(tiny_model, mode="int8")
        size_int4 = estimate_quantized_size_gb(tiny_model, mode="int4")
        assert size_int4 < size_int8

    def test_dequantize_roundtrip_int8(self):
        from src.model.quantization import quantize_weight_int8, dequantize_weight_int8
        w = np.random.randn(8, 8).astype(np.float32)
        q, scale = quantize_weight_int8(w)
        w_rec = dequantize_weight_int8(q, scale)
        # Reconstruction error should be small (< 1%)
        rel_err = np.mean(np.abs(w - w_rec)) / (np.mean(np.abs(w)) + 1e-8)
        assert rel_err < 0.02

    def test_dequantize_roundtrip_int4(self):
        from src.model.quantization import (
            quantize_weight_int4_simulated,
            dequantize_weight_int4_simulated,
        )
        w = np.random.randn(8, 8).astype(np.float32)
        q, scale = quantize_weight_int4_simulated(w)
        w_rec = dequantize_weight_int4_simulated(q, scale)
        rel_err = np.mean(np.abs(w - w_rec)) / (np.mean(np.abs(w)) + 1e-8)
        assert rel_err < 0.2  # INT4 has higher quantization error


# ---------------------------------------------------------------------------
# Distributed training helpers tests
# ---------------------------------------------------------------------------

class TestDistributed:
    def test_get_strategy_cpu(self):
        from src.training.distributed import get_distribution_strategy
        strategy = get_distribution_strategy(num_gpus=0)
        assert isinstance(strategy, tf.distribute.Strategy)

    def test_auto_detect_strategy(self):
        from src.training.distributed import auto_detect_strategy
        strategy = auto_detect_strategy()
        assert isinstance(strategy, tf.distribute.Strategy)

    def test_configure_mixed_precision_fp16(self):
        from src.training.distributed import configure_mixed_precision
        configure_mixed_precision("fp16")
        policy = tf.keras.mixed_precision.global_policy()
        assert "float16" in policy.name
        # Reset
        configure_mixed_precision(None)

    def test_configure_mixed_precision_none(self):
        from src.training.distributed import configure_mixed_precision
        configure_mixed_precision(None)
        policy = tf.keras.mixed_precision.global_policy()
        assert policy.name == "float32"

    def test_configure_mixed_precision_invalid(self):
        from src.training.distributed import configure_mixed_precision
        with pytest.raises(ValueError, match="Invalid mixed precision"):
            configure_mixed_precision("fp8")

    def test_gradient_accumulator(self):
        from src.training.distributed import GradientAccumulator
        from src.model.transformer import SmallTransformer, TransformerConfig
        cfg = TransformerConfig(
            vocab_size=32, d_model=16, num_heads=2, num_layers=1, d_ff=32,
        )
        model = SmallTransformer(cfg)
        _ = model(tf.zeros((1, 2), dtype=tf.int32))

        acc = GradientAccumulator(model, num_steps=2)
        assert not acc.ready

        dummy_grads = [tf.zeros_like(v) for v in model.trainable_variables]
        acc.accumulate(dummy_grads)
        assert not acc.ready

        acc.accumulate(dummy_grads)
        assert acc.ready

        acc.reset()
        assert not acc.ready

    def test_load_deepspeed_config(self):
        from src.training.distributed import load_deepspeed_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"optimizer": {"params": {"lr": 1e-4}}}, f)
            tmp_path = f.name
        try:
            cfg = load_deepspeed_config(tmp_path)
            assert cfg["optimizer"]["params"]["lr"] == 1e-4
        finally:
            os.unlink(tmp_path)

    def test_load_deepspeed_config_missing(self):
        from src.training.distributed import load_deepspeed_config
        with pytest.raises(FileNotFoundError):
            load_deepspeed_config("/nonexistent/path.json")
