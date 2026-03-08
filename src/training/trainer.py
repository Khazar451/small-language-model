"""
Training loop and pipeline for small language models.

This module provides a Trainer class that handles the full training pipeline
including gradient accumulation, checkpointing, logging, and early stopping.
"""

import logging
import os
import time
import json
from typing import Optional, Dict, Any, List, Callable

import numpy as np
import tensorflow as tf

from src.training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Full training pipeline for language models.

    Handles training loops, validation, checkpointing, learning rate scheduling,
    gradient accumulation, and early stopping.

    Args:
        model: TensorFlow/Keras model to train.
        optimizer: Optimizer (or a string name for automatic creation).
        train_dataset: Training tf.data.Dataset.
        val_dataset: Optional validation tf.data.Dataset.
        num_epochs: Number of training epochs.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        max_grad_norm: Maximum gradient norm for clipping.
        output_dir: Directory to save checkpoints and logs.
        save_steps: Save checkpoint every N steps.
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Log metrics every N steps.
        eval_steps: Run validation every N steps (0 = only at end of epoch).
        early_stopping_patience: Number of evaluations without improvement before stopping.
        metric_for_best_model: Metric to use for model selection.
        greater_is_better: Whether higher metric values are better.
        callbacks: Additional Keras callbacks.

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer="adamw",
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     num_epochs=3,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: Any,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs/checkpoints",
        save_steps: int = 500,
        save_total_limit: int = 3,
        logging_steps: int = 100,
        eval_steps: int = 0,
        early_stopping_patience: int = 0,
        metric_for_best_model: str = "val_loss",
        greater_is_better: bool = False,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

        # Build optimizer
        if isinstance(optimizer, str):
            self.optimizer = self._build_optimizer(optimizer)
        else:
            self.optimizer = optimizer

        self.metrics_tracker = MetricsTracker()
        self.callbacks = callbacks or []
        self.global_step = 0
        self._checkpoint_paths: List[str] = []

        os.makedirs(output_dir, exist_ok=True)

    def _build_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> tf.keras.optimizers.Optimizer:
        """Build an optimizer by name.

        Args:
            optimizer_name: Name of the optimizer ('adam' or 'adamw').
            learning_rate: Initial learning rate.
            weight_decay: Weight decay coefficient (for AdamW).
            beta1: First Adam moment decay.
            beta2: Second Adam moment decay.
            epsilon: Numerical stability epsilon.

        Returns:
            Configured optimizer.
        """
        if optimizer_name.lower() == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta_1=beta1,
                beta_2=beta2,
                epsilon=epsilon,
            )
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
        )

    @tf.function
    def _train_step(
        self,
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor],
        labels: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Perform a single training step.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Optional labels.

        Returns:
            Dictionary with 'loss' and other metrics.
        """
        with tf.GradientTape() as tape:
            outputs = self.model.compute_loss(
                input_ids=input_ids,
                labels=labels if labels is not None else input_ids,
                attention_mask=attention_mask,
                training=True,
            )
            loss = outputs["loss"] / tf.cast(self.gradient_accumulation_steps, tf.float32)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [
            tf.clip_by_norm(g, self.max_grad_norm) if g is not None else g
            for g in gradients
        ]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return outputs

    def _accumulate_gradients(
        self,
        batches: List[Any],
    ) -> Dict[str, float]:
        """Accumulate gradients over multiple batches before updating.

        Args:
            batches: List of batch tuples to accumulate over.

        Returns:
            Dictionary with averaged metrics.
        """
        accumulated_grads = None  # Initialized lazily after first forward pass
        total_loss = 0.0

        for batch in batches:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
            else:
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")

            with tf.GradientTape() as tape:
                outputs = self.model.compute_loss(
                    input_ids=input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                    training=True,
                )
                scaled_loss = outputs["loss"] / len(batches)

            grads = tape.gradient(scaled_loss, self.model.trainable_variables)

            if accumulated_grads is None:
                # Initialize after first forward pass so model is built
                accumulated_grads = [
                    tf.convert_to_tensor(g) if g is not None else tf.zeros_like(v)
                    for g, v in zip(grads, self.model.trainable_variables)
                ]
            else:
                accumulated_grads = [
                    acc + (tf.convert_to_tensor(g) if g is not None else tf.zeros_like(acc))
                    for acc, g in zip(accumulated_grads, grads)
                ]
            total_loss += float(outputs["loss"])

        if accumulated_grads is not None:
            # Clip and apply accumulated gradients
            clipped_grads = [
                tf.clip_by_norm(g, self.max_grad_norm)
                for g in accumulated_grads
            ]
            self.optimizer.apply_gradients(
                zip(clipped_grads, self.model.trainable_variables)
            )

        return {"loss": total_loss / len(batches)}

    def _eval_step(
        self,
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Perform a single evaluation step.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Dictionary with loss and other metrics.
        """
        outputs = self.model.compute_loss(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
            training=False,
        )
        return outputs

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation dataset.

        Returns:
            Dictionary of average metrics over the validation set.
        """
        if self.val_dataset is None:
            return {}

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataset:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
            else:
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")

            outputs = self._eval_step(input_ids, attention_mask)
            total_loss += float(outputs["loss"])
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = float(np.exp(min(avg_loss, 20)))

        metrics = {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }
        logger.info("Validation - Loss: %.4f | Perplexity: %.2f", avg_loss, perplexity)
        return metrics

    def _save_checkpoint(self, step: int) -> None:
        """Save a model checkpoint.

        Args:
            step: Current training step.
        """
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(ckpt_dir)
        else:
            self.model.save_weights(os.path.join(ckpt_dir, "weights.weights.h5"))

        self._checkpoint_paths.append(ckpt_dir)

        # Enforce checkpoint limit
        while len(self._checkpoint_paths) > self.save_total_limit:
            old_ckpt = self._checkpoint_paths.pop(0)
            import shutil
            if os.path.exists(old_ckpt):
                shutil.rmtree(old_ckpt)
                logger.debug("Removed old checkpoint: %s", old_ckpt)

        logger.info("Saved checkpoint to %s", ckpt_dir)

    def _save_best_model(self) -> None:
        """Save the best model to the output directory."""
        best_dir = os.path.join(self.output_dir, "best_model")
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(best_dir)
        else:
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_weights(os.path.join(best_dir, "weights.weights.h5"))
        logger.info("Saved best model to %s", best_dir)

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            Dictionary of training history with loss and metric values.
        """
        logger.info(
            "Starting training for %d epochs, %d gradient accumulation steps",
            self.num_epochs, self.gradient_accumulation_steps,
        )

        best_metric = float("inf") if not self.greater_is_better else float("-inf")
        no_improvement_count = 0
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
        }

        for epoch in range(1, self.num_epochs + 1):
            logger.info("Epoch %d/%d", epoch, self.num_epochs)
            epoch_start = time.time()
            epoch_loss = 0.0
            num_steps = 0
            epoch_tokens = 0

            accum_batches = []

            for batch in self.train_dataset:
                accum_batches.append(batch)

                # Count tokens for throughput tracking
                if isinstance(batch, (tuple, list)):
                    batch_ids = batch[0]
                    batch_mask = batch[1] if len(batch) > 1 else None
                else:
                    batch_ids = batch.get("input_ids")
                    batch_mask = batch.get("attention_mask")

                if batch_mask is not None:
                    epoch_tokens += int(tf.reduce_sum(batch_mask).numpy())
                else:
                    epoch_tokens += int(tf.size(batch_ids).numpy())

                if len(accum_batches) < self.gradient_accumulation_steps:
                    continue

                # Perform gradient accumulation step
                step_metrics = self._accumulate_gradients(accum_batches)
                accum_batches = []

                self.global_step += 1
                epoch_loss += step_metrics["loss"]
                num_steps += 1

                self.metrics_tracker.update("train_loss", step_metrics["loss"])

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_steps
                    elapsed = time.time() - epoch_start
                    tokens_per_sec = epoch_tokens / max(elapsed, 1e-9)
                    logger.info(
                        "Step %d | Loss: %.4f | LR: %.2e | Tokens/s: %.0f",
                        self.global_step,
                        avg_loss,
                        float(self.optimizer.learning_rate),
                        tokens_per_sec,
                    )

                # Mid-epoch evaluation
                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate()
                    self.metrics_tracker.update_dict(val_metrics)

                    metric_val = val_metrics.get(self.metric_for_best_model, float("inf"))
                    if self._is_better(metric_val, best_metric):
                        best_metric = metric_val
                        no_improvement_count = 0
                        self._save_best_model()
                    else:
                        no_improvement_count += 1

                    if (
                        self.early_stopping_patience > 0
                        and no_improvement_count >= self.early_stopping_patience
                    ):
                        logger.info("Early stopping triggered at step %d", self.global_step)
                        return history

                # Checkpoint saving
                if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.global_step)

            # Handle leftover batches
            if accum_batches:
                step_metrics = self._accumulate_gradients(accum_batches)
                epoch_loss += step_metrics["loss"]
                num_steps += 1

            avg_epoch_loss = epoch_loss / max(num_steps, 1)
            elapsed = time.time() - epoch_start
            tokens_per_sec = epoch_tokens / max(elapsed, 1e-9)
            logger.info(
                "Epoch %d done in %.1fs | Avg Loss: %.4f | Tokens/s: %.0f",
                epoch, elapsed, avg_epoch_loss, tokens_per_sec,
            )
            history["train_loss"].append(avg_epoch_loss)

            # End-of-epoch evaluation
            val_metrics = self.evaluate()
            if val_metrics:
                history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                history["val_perplexity"].append(val_metrics.get("val_perplexity", 0.0))

                metric_val = val_metrics.get(self.metric_for_best_model, float("inf"))
                if self._is_better(metric_val, best_metric):
                    best_metric = metric_val
                    no_improvement_count = 0
                    self._save_best_model()
                else:
                    no_improvement_count += 1

                if (
                    self.early_stopping_patience > 0
                    and no_improvement_count >= self.early_stopping_patience
                ):
                    logger.info("Early stopping triggered after epoch %d", epoch)
                    break

        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

        return history

    def _is_better(self, new_val: float, best_val: float) -> bool:
        """Check if a metric value is better than the current best.

        Args:
            new_val: New metric value.
            best_val: Current best metric value.

        Returns:
            True if new_val is better.
        """
        if self.greater_is_better:
            return new_val > best_val
        return new_val < best_val
