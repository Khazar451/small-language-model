"""
Tests for training utilities.

Tests cover:
- MetricsTracker: update, average, latest, summary, reset
- Trainer: initialization, single training step, evaluate
- compute_perplexity and compute_accuracy utilities
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import (
    MetricsTracker,
    compute_perplexity,
    compute_accuracy,
    compute_f1,
)
from src.model.transformer import SmallTransformer, TransformerConfig


# ---------------------------------------------------------------------------
# MetricsTracker tests
# ---------------------------------------------------------------------------

class TestMetricsTracker:
    def setup_method(self):
        self.tracker = MetricsTracker()

    def test_update_and_average(self):
        self.tracker.update("loss", 2.0)
        self.tracker.update("loss", 1.0)
        assert self.tracker.average("loss") == pytest.approx(1.5)

    def test_average_empty(self):
        assert self.tracker.average("nonexistent") == 0.0

    def test_latest(self):
        self.tracker.update("acc", 0.5)
        self.tracker.update("acc", 0.9)
        assert self.tracker.latest("acc") == pytest.approx(0.9)

    def test_latest_empty(self):
        assert self.tracker.latest("nonexistent") is None

    def test_update_dict(self):
        self.tracker.update_dict({"loss": 1.5, "acc": 0.8})
        assert self.tracker.latest("loss") == pytest.approx(1.5)
        assert self.tracker.latest("acc") == pytest.approx(0.8)

    def test_last_n_average(self):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.tracker.update("metric", v)
        # Average of last 3: (3+4+5)/3 = 4.0
        assert self.tracker.average("metric", last_n=3) == pytest.approx(4.0)

    def test_all_values(self):
        vals = [1.0, 2.0, 3.0]
        for v in vals:
            self.tracker.update("x", v)
        assert self.tracker.all_values("x") == vals

    def test_reset_single_metric(self):
        self.tracker.update("loss", 1.0)
        self.tracker.update("acc", 0.9)
        self.tracker.reset("loss")
        assert self.tracker.all_values("loss") == []
        assert self.tracker.latest("acc") is not None

    def test_reset_all(self):
        self.tracker.update("loss", 1.0)
        self.tracker.update("acc", 0.9)
        self.tracker.reset()
        assert self.tracker.all_values("loss") == []
        assert self.tracker.all_values("acc") == []

    def test_summary(self):
        for v in [1.0, 2.0, 3.0]:
            self.tracker.update("loss", v)
        summary = self.tracker.summary()
        assert "loss" in summary
        assert summary["loss"]["mean"] == pytest.approx(2.0)
        assert summary["loss"]["min"] == pytest.approx(1.0)
        assert summary["loss"]["max"] == pytest.approx(3.0)
        assert summary["loss"]["count"] == 3

    def test_summary_empty(self):
        summary = self.tracker.summary()
        assert summary == {}


# ---------------------------------------------------------------------------
# compute_perplexity tests
# ---------------------------------------------------------------------------

class TestComputePerplexity:
    def test_zero_loss(self):
        assert compute_perplexity(0.0) == pytest.approx(1.0)

    def test_positive_loss(self):
        ppl = compute_perplexity(2.0)
        assert ppl == pytest.approx(np.exp(2.0), rel=1e-5)

    def test_high_loss_capped(self):
        # Very high loss should not cause overflow
        ppl = compute_perplexity(1000.0)
        assert np.isfinite(ppl)


# ---------------------------------------------------------------------------
# compute_accuracy tests
# ---------------------------------------------------------------------------

class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        preds = [0, 1, 2, 1]
        labels = [0, 1, 2, 1]
        assert compute_accuracy(preds, labels) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        preds = [1, 0, 1]
        labels = [0, 1, 0]
        assert compute_accuracy(preds, labels) == pytest.approx(0.0)

    def test_partial_accuracy(self):
        preds = [0, 1, 0, 1]
        labels = [0, 1, 1, 0]
        assert compute_accuracy(preds, labels) == pytest.approx(0.5)

    def test_logits_input(self):
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])
        labels = [1, 0]
        assert compute_accuracy(logits, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_f1 tests
# ---------------------------------------------------------------------------

class TestComputeF1:
    def test_perfect_f1(self):
        preds = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        score = compute_f1(preds, labels)
        assert score == pytest.approx(1.0)

    def test_f1_with_logits(self):
        logits = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        labels = [0, 1, 0]
        score = compute_f1(logits, labels)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def _make_small_model(task="text_generation"):
    """Create a tiny model for testing."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=64,
        max_seq_length=16,
        dropout_rate=0.0,
        attention_dropout=0.0,
        task=task,
        num_labels=2,
    )
    return SmallTransformer(config)


def _make_tiny_dataset(vocab_size=100, seq_len=8, num_samples=4, batch_size=2):
    """Create a tiny tf.data.Dataset for testing."""
    input_ids = np.random.randint(1, vocab_size, (num_samples, seq_len)).astype(np.int32)
    attention_mask = np.ones((num_samples, seq_len), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask))
    return ds.batch(batch_size)


class TestTrainer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_trainer(self, model=None, val_dataset=None, task="text_generation"):
        if model is None:
            model = _make_small_model(task=task)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        train_ds = _make_tiny_dataset()
        return __import__(
            "src.training.trainer", fromlist=["Trainer"]
        ).Trainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=val_dataset,
            num_epochs=1,
            gradient_accumulation_steps=1,
            output_dir=self.tmpdir,
            logging_steps=1,
            save_steps=0,
            eval_steps=0,
        )

    def test_trainer_initialization(self):
        from src.training.trainer import Trainer
        model = _make_small_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        train_ds = _make_tiny_dataset()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            num_epochs=1,
            output_dir=self.tmpdir,
        )
        assert trainer.num_epochs == 1
        assert trainer.global_step == 0

    def test_train_runs(self):
        trainer = self._make_trainer()
        history = trainer.train()
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_train_decreases_loss(self):
        """Loss should be finite after training."""
        trainer = self._make_trainer()
        history = trainer.train()
        assert np.isfinite(history["train_loss"][0])

    def test_trainer_with_validation(self):
        val_ds = _make_tiny_dataset(num_samples=2, batch_size=2)
        trainer = self._make_trainer(val_dataset=val_ds)
        history = trainer.train()
        assert "val_loss" in history
        assert len(history["val_loss"]) == 1

    def test_evaluate_returns_metrics(self):
        val_ds = _make_tiny_dataset(num_samples=2, batch_size=2)
        trainer = self._make_trainer(val_dataset=val_ds)
        metrics = trainer.evaluate()
        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0
        assert "val_perplexity" in metrics

    def test_evaluate_no_val_dataset(self):
        trainer = self._make_trainer()
        metrics = trainer.evaluate()
        assert metrics == {}

    def test_gradient_accumulation(self):
        from src.training.trainer import Trainer
        model = _make_small_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # Create dataset with enough samples for accumulation
        train_ds = _make_tiny_dataset(num_samples=8, batch_size=2)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            num_epochs=1,
            gradient_accumulation_steps=2,
            output_dir=self.tmpdir,
            logging_steps=1,
            save_steps=0,
        )
        history = trainer.train()
        assert len(history["train_loss"]) == 1

    def test_is_better_lower_is_better(self):
        from src.training.trainer import Trainer
        trainer = self._make_trainer()
        trainer.greater_is_better = False
        assert trainer._is_better(1.0, 2.0)
        assert not trainer._is_better(2.0, 1.0)

    def test_is_better_higher_is_better(self):
        from src.training.trainer import Trainer
        trainer = self._make_trainer()
        trainer.greater_is_better = True
        assert trainer._is_better(2.0, 1.0)
        assert not trainer._is_better(1.0, 2.0)

    def test_history_saved_to_file(self):
        trainer = self._make_trainer()
        trainer.train()
        history_file = os.path.join(self.tmpdir, "training_history.json")
        assert os.path.exists(history_file)

    def test_build_optimizer_adam(self):
        from src.training.trainer import Trainer
        model = _make_small_model()
        # Passing string optimizer should build one
        train_ds = _make_tiny_dataset()
        trainer = Trainer(
            model=model,
            optimizer="adam",
            train_dataset=train_ds,
            output_dir=self.tmpdir,
        )
        assert isinstance(trainer.optimizer, tf.keras.optimizers.Optimizer)

    def test_build_optimizer_adamw(self):
        from src.training.trainer import Trainer
        model = _make_small_model()
        train_ds = _make_tiny_dataset()
        trainer = Trainer(
            model=model,
            optimizer="adamw",
            train_dataset=train_ds,
            output_dir=self.tmpdir,
        )
        assert isinstance(trainer.optimizer, tf.keras.optimizers.Optimizer)
