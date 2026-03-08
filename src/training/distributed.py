"""
Distributed training utilities for multi-GPU and multi-node setups.

Wraps TensorFlow's distribution strategies and provides a unified
``DistributedTrainer`` interface that mirrors the single-device ``Trainer``.
"""

import logging
import os
from typing import Any, List, Optional

import tensorflow as tf

from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

# Supported distribution strategies
_STRATEGIES = ("mirrored", "tpu", "parameter_server", "one_device")


class DistributedTrainer:
    """Multi-GPU / multi-node training wrapper.

    Selects the appropriate TensorFlow distribution strategy and delegates
    model creation and training to the underlying :class:`~src.training.trainer.Trainer`.

    Args:
        model_fn: Zero-argument callable that returns a ``tf.keras.Model``.
            Called *inside* the strategy scope so that variables are
            distributed correctly.  Mutually exclusive with *model*.
        model: Pre-built model.  If provided, the model should have been
            created inside the target strategy scope.
        strategy: Distribution strategy name.  One of ``"mirrored"``,
            ``"tpu"``, ``"parameter_server"``, or ``"one_device"``.
        num_gpus: Number of GPUs to use with the ``"mirrored"`` strategy.
            Pass ``0`` to use all available GPUs.
        tpu_address: TPU address (only used with ``strategy="tpu"``).
        train_dataset: Training ``tf.data.Dataset``.
        val_dataset: Optional validation ``tf.data.Dataset``.
        optimizer: Optimizer instance or string name.
        gradient_accumulation_steps: Steps to accumulate gradients over.
        output_dir: Directory for checkpoints and logs.
        **trainer_kwargs: Additional keyword arguments forwarded to
            :class:`~src.training.trainer.Trainer`.

    Example:
        >>> dist_trainer = DistributedTrainer(
        ...     model_fn=lambda: SmallTransformer(config),
        ...     strategy="mirrored",
        ...     num_gpus=4,
        ...     train_dataset=train_ds,
        ...     optimizer="adamw",
        ... )
        >>> dist_trainer.train(num_epochs=3)
    """

    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        model_fn: Optional[Any] = None,
        model: Optional[tf.keras.Model] = None,
        strategy: str = "mirrored",
        num_gpus: int = 0,
        tpu_address: Optional[str] = None,
        val_dataset: Optional[tf.data.Dataset] = None,
        optimizer: Any = "adamw",
        gradient_accumulation_steps: int = 1,
        output_dir: str = "outputs/distributed",
        **trainer_kwargs,
    ):
        if model_fn is None and model is None:
            raise ValueError("Provide either model_fn or model.")
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_STRATEGIES}, got {strategy!r}"
            )

        self.strategy_name = strategy
        self.strategy = self._build_strategy(strategy, num_gpus, tpu_address)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.trainer_kwargs = trainer_kwargs

        with self.strategy.scope():
            if model_fn is not None:
                self._model = model_fn()
            else:
                self._model = model

        logger.info(
            "DistributedTrainer: strategy=%s, replicas=%d",
            strategy,
            self.strategy.num_replicas_in_sync,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strategy(
        name: str,
        num_gpus: int,
        tpu_address: Optional[str],
    ) -> tf.distribute.Strategy:
        """Construct the appropriate TF distribution strategy."""
        if name == "mirrored":
            gpus = tf.config.list_physical_devices("GPU")
            if num_gpus > 0:
                gpus = gpus[:num_gpus]
            if gpus:
                devices = [g.name for g in gpus]
                return tf.distribute.MirroredStrategy(devices=devices)
            else:
                logger.warning("No GPUs found, falling back to MirroredStrategy with CPU.")
                return tf.distribute.MirroredStrategy()

        if name == "tpu":
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)

        if name == "parameter_server":
            return tf.distribute.experimental.ParameterServerStrategy(
                tf.distribute.cluster_resolver.TFConfigClusterResolver()
            )

        # one_device or fallback
        return tf.distribute.OneDeviceStrategy(device="/gpu:0" if num_gpus else "/cpu:0")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> Any:
        """Train the model using the configured distribution strategy.

        Args:
            num_epochs: Number of epochs.  Overrides the value in
                *trainer_kwargs* if provided.

        Returns:
            Training history dict from the underlying ``Trainer``.
        """
        kwargs = dict(self.trainer_kwargs)
        if num_epochs is not None:
            kwargs["num_epochs"] = num_epochs

        trainer = Trainer(
            model=self._model,
            optimizer=self.optimizer,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            output_dir=self.output_dir,
            **kwargs,
        )
        return trainer.train()

    @property
    def model(self) -> tf.keras.Model:
        """The distributed model."""
        return self._model

    @property
    def num_replicas(self) -> int:
        """Number of active replicas (GPUs/TPU cores)."""
        return self.strategy.num_replicas_in_sync
