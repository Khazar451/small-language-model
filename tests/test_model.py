"""
Tests for transformer model components.

Tests cover:
- TransformerConfig creation and serialization
- MultiHeadAttention forward pass
- PositionwiseFeedForward forward pass
- TransformerBlock forward pass
- SmallTransformer full forward pass for all task types
- Causal mask and padding mask creation
- Model save/load
- Parameter counting
"""

import os
import sys
import tempfile
import json

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import (
    SmallTransformer,
    TransformerConfig,
    MultiHeadAttention,
    PositionwiseFeedForward,
    TransformerBlock,
    create_causal_mask,
    create_padding_mask,
    get_sinusoidal_encoding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CONFIG = TransformerConfig(
    vocab_size=100,
    d_model=64,
    num_heads=4,
    num_layers=2,
    d_ff=128,
    max_seq_length=32,
    dropout_rate=0.0,
    attention_dropout=0.0,
    task="text_generation",
)

BATCH_SIZE = 2
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# TransformerConfig tests
# ---------------------------------------------------------------------------

class TestTransformerConfig:
    def test_default_config(self):
        config = TransformerConfig()
        assert config.vocab_size == 50257
        assert config.d_model == 768
        assert config.num_heads == 12

    def test_custom_config(self):
        config = TransformerConfig(vocab_size=100, d_model=64, num_heads=4)
        assert config.vocab_size == 100
        assert config.d_model == 64

    def test_from_dict(self):
        d = {"vocab_size": 200, "d_model": 128, "num_heads": 4, "num_layers": 2,
             "d_ff": 256, "task": "sentiment_analysis"}
        config = TransformerConfig.from_dict(d)
        assert config.vocab_size == 200
        assert config.task == "sentiment_analysis"

    def test_to_dict(self):
        config = TransformerConfig(vocab_size=50, d_model=32, num_heads=2)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["vocab_size"] == 50
        assert "task" in d

    def test_roundtrip(self):
        config = TransformerConfig(vocab_size=300, d_model=128, num_heads=8)
        restored = TransformerConfig.from_dict(config.to_dict())
        assert restored.vocab_size == config.vocab_size
        assert restored.d_model == config.d_model

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict should silently ignore unknown keys."""
        d = {"vocab_size": 100, "d_model": 64, "num_heads": 4, "unknown_key": "value"}
        # Should not raise
        config = TransformerConfig.from_dict(d)
        assert config.vocab_size == 100


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------

class TestMasks:
    def test_causal_mask_shape(self):
        mask = create_causal_mask(10)
        assert mask.shape == (1, 1, 10, 10)

    def test_causal_mask_upper_triangular(self):
        mask = create_causal_mask(5).numpy()[0, 0]
        # Upper triangle (excluding diagonal) should be 1
        for i in range(5):
            for j in range(5):
                if j > i:
                    assert mask[i, j] == 1.0, f"mask[{i},{j}] should be 1"
                else:
                    assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"

    def test_padding_mask_shape(self):
        token_ids = tf.constant([[1, 2, 0, 0], [3, 0, 0, 0]])
        mask = create_padding_mask(token_ids, pad_token_id=0)
        assert mask.shape == (2, 1, 1, 4)

    def test_padding_mask_values(self):
        token_ids = tf.constant([[1, 2, 0]])
        mask = create_padding_mask(token_ids, pad_token_id=0).numpy()[0, 0, 0]
        assert mask[0] == 0.0  # non-pad
        assert mask[1] == 0.0  # non-pad
        assert mask[2] == 1.0  # pad

    def test_sinusoidal_encoding_shape(self):
        enc = get_sinusoidal_encoding(16, 64)
        assert enc.shape == (1, 16, 64)


# ---------------------------------------------------------------------------
# MultiHeadAttention tests
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=64, num_heads=4)
        q = tf.random.normal((BATCH_SIZE, SEQ_LEN, 64))
        output, weights = attn(q, q, q)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)
        assert weights.shape == (BATCH_SIZE, 4, SEQ_LEN, SEQ_LEN)

    def test_with_mask(self):
        attn = MultiHeadAttention(d_model=64, num_heads=4)
        q = tf.random.normal((BATCH_SIZE, SEQ_LEN, 64))
        mask = create_causal_mask(SEQ_LEN)
        output, _ = attn(q, q, q, mask=mask)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)

    def test_invalid_d_model_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)  # Not divisible

    def test_get_config(self):
        attn = MultiHeadAttention(d_model=64, num_heads=4, name="test_attn")
        config = attn.get_config()
        assert config["d_model"] == 64
        assert config["num_heads"] == 4


# ---------------------------------------------------------------------------
# PositionwiseFeedForward tests
# ---------------------------------------------------------------------------

class TestPositionwiseFeedForward:
    def test_output_shape(self):
        ffn = PositionwiseFeedForward(d_model=64, d_ff=256)
        x = tf.random.normal((BATCH_SIZE, SEQ_LEN, 64))
        output = ffn(x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)

    def test_relu_activation(self):
        ffn = PositionwiseFeedForward(d_model=32, d_ff=64, activation="relu")
        x = tf.random.normal((2, 4, 32))
        output = ffn(x)
        assert output.shape == (2, 4, 32)


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
        x = tf.random.normal((BATCH_SIZE, SEQ_LEN, 64))
        output, weights = block(x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)
        assert weights.shape == (BATCH_SIZE, 4, SEQ_LEN, SEQ_LEN)

    def test_with_causal_mask(self):
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
        x = tf.random.normal((BATCH_SIZE, SEQ_LEN, 64))
        mask = create_causal_mask(SEQ_LEN)
        output, _ = block(x, mask=mask)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)


# ---------------------------------------------------------------------------
# SmallTransformer tests
# ---------------------------------------------------------------------------

class TestSmallTransformer:
    def _make_model(self, task="text_generation", num_labels=2):
        config = TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_seq_length=32,
            dropout_rate=0.0,
            attention_dropout=0.0,
            task=task,
            num_labels=num_labels,
        )
        return SmallTransformer(config)

    def _make_inputs(self):
        return tf.constant(
            np.random.randint(1, 100, (BATCH_SIZE, SEQ_LEN)), dtype=tf.int32
        )

    def test_text_generation_output(self):
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (BATCH_SIZE, SEQ_LEN, 100)
        assert "hidden_states" in outputs

    def test_sentiment_analysis_output(self):
        model = self._make_model("sentiment_analysis", num_labels=2)
        input_ids = self._make_inputs()
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (BATCH_SIZE, 2)

    def test_sequence_classification_output(self):
        model = self._make_model("sequence_classification", num_labels=3)
        input_ids = self._make_inputs()
        outputs = model(input_ids)
        assert outputs["logits"].shape == (BATCH_SIZE, 3)

    def test_question_answering_output(self):
        model = self._make_model("question_answering")
        input_ids = self._make_inputs()
        outputs = model(input_ids)
        assert "start_logits" in outputs
        assert "end_logits" in outputs
        assert outputs["start_logits"].shape == (BATCH_SIZE, SEQ_LEN)
        assert outputs["end_logits"].shape == (BATCH_SIZE, SEQ_LEN)

    def test_with_attention_mask(self):
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        attention_mask = tf.ones((BATCH_SIZE, SEQ_LEN), dtype=tf.int32)
        outputs = model(input_ids, attention_mask=attention_mask)
        assert "logits" in outputs

    def test_sinusoidal_encoding(self):
        config = TransformerConfig(
            vocab_size=100, d_model=64, num_heads=4, num_layers=2,
            d_ff=128, positional_encoding="sinusoidal", dropout_rate=0.0
        )
        model = SmallTransformer(config)
        input_ids = self._make_inputs()
        outputs = model(input_ids)
        assert "logits" in outputs

    def test_return_attention_weights(self):
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        outputs = model(input_ids, return_attention_weights=True)
        assert "attention_weights" in outputs
        assert len(outputs["attention_weights"]) == 2  # num_layers

    def test_compute_loss_text_generation(self):
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        outputs = model.compute_loss(input_ids=input_ids, labels=input_ids)
        assert "loss" in outputs
        loss_val = float(outputs["loss"])
        assert loss_val > 0
        assert not np.isnan(loss_val)

    def test_compute_loss_classification(self):
        model = self._make_model("sentiment_analysis", num_labels=2)
        input_ids = self._make_inputs()
        labels = tf.zeros((BATCH_SIZE,), dtype=tf.int32)
        outputs = model.compute_loss(input_ids=input_ids, labels=labels)
        assert "loss" in outputs
        assert float(outputs["loss"]) >= 0

    def test_compute_loss_qa(self):
        model = self._make_model("question_answering")
        input_ids = self._make_inputs()
        start = tf.zeros((BATCH_SIZE,), dtype=tf.int32)
        end = tf.ones((BATCH_SIZE,), dtype=tf.int32)
        outputs = model.compute_loss(
            input_ids=input_ids, start_positions=start, end_positions=end
        )
        assert "loss" in outputs

    def test_count_parameters(self):
        model = self._make_model("text_generation")
        # Build model with a forward pass
        input_ids = self._make_inputs()
        model(input_ids)
        count = model.count_parameters()
        assert count > 0

    def test_save_load(self):
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        original_output = model(input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "config.json"))

            # Load and compare outputs
            loaded_model = SmallTransformer.load_pretrained(tmpdir)
            loaded_output = loaded_model(input_ids)

            np.testing.assert_allclose(
                original_output["logits"].numpy(),
                loaded_output["logits"].numpy(),
                atol=1e-5,
            )

    def test_from_config(self):
        model = self._make_model("text_generation")
        config_dict = model.get_config()
        restored = SmallTransformer.from_config(config_dict)
        assert restored.config.vocab_size == model.config.vocab_size
        assert restored.config.d_model == model.config.d_model

    def test_no_nan_in_outputs(self):
        """Ensure no NaN values in model outputs."""
        model = self._make_model("text_generation")
        input_ids = self._make_inputs()
        outputs = model(input_ids, training=False)
        assert not np.any(np.isnan(outputs["logits"].numpy()))
        assert not np.any(np.isinf(outputs["logits"].numpy()))
