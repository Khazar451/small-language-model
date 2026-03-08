"""
Distributed training utilities for large language models.

Provides helpers for:
- Multi-GPU training with tf.distribute strategies
- Zero Redundancy Optimizer (ZeRO)-style gradient sharding utilities
- Automatic device detection and strategy selection
- DeepSpeed-style configuration parsing
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List

import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def get_distribution_strategy(
    num_gpus: int = 1,
    use_tpu: bool = False,
    tpu_address: str = "",
) -> tf.distribute.Strategy:
    """Create the appropriate tf.distribute strategy for the hardware.

    Args:
        num_gpus: Number of GPUs to use.  0 = CPU only.
        use_tpu: Whether to use a TPU.
        tpu_address: TPU address (used only when use_tpu=True).

    Returns:
        A tf.distribute.Strategy instance.
    """
    if use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info("Using TPU strategy (%s)", tpu_address or "local")
        return strategy

    gpus = tf.config.list_physical_devices("GPU")
    available = len(gpus)
    requested = min(num_gpus, available) if num_gpus > 0 else 0

    if requested >= 2:
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/GPU:{i}" for i in range(requested)]
        )
        logger.info("Using MirroredStrategy across %d GPUs", requested)
        return strategy

    if requested == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        logger.info("Using single GPU strategy (GPU:0)")
        return strategy

    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
    logger.info("No GPU detected — using CPU strategy")
    return strategy


def auto_detect_strategy() -> tf.distribute.Strategy:
    """Automatically detect and return the best available strategy.

    Prefers: TPU > multi-GPU > single-GPU > CPU.

    Returns:
        A tf.distribute.Strategy instance.
    """
    # Check for TPU
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info("Auto-detected TPU — using TPUStrategy")
        return strategy
    except (ValueError, RuntimeError):
        pass

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info("Auto-detected %d GPUs — using MirroredStrategy", len(gpus))
        return strategy

    if len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        logger.info("Auto-detected 1 GPU — using OneDeviceStrategy")
        return strategy

    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
    logger.info("No accelerator detected — using CPU strategy")
    return strategy


# ---------------------------------------------------------------------------
# Mixed precision helpers
# ---------------------------------------------------------------------------

def configure_mixed_precision(mode: Optional[str] = None) -> None:
    """Configure Keras mixed precision globally.

    Args:
        mode: One of ``'fp16'``, ``'bf16'``, or ``None`` (disabled).
            - ``'fp16'``:  Uses float16 for layer computations and float32 for
              variables.  Best suited for NVIDIA GPUs with Tensor Cores.
            - ``'bf16'``: Uses bfloat16.  Recommended for TPUs and newer NVIDIA
              Ampere GPUs.
            - ``None``: Resets to default float32 policy.
    """
    policy_map = {
        "fp16": "mixed_float16",
        "bf16": "mixed_bfloat16",
    }
    if mode is None:
        tf.keras.mixed_precision.set_global_policy("float32")
        logger.info("Mixed precision disabled (float32)")
        return

    policy_name = policy_map.get(mode.lower())
    if policy_name is None:
        raise ValueError(
            f"Invalid mixed precision mode '{mode}'. "
            f"Choose from: {list(policy_map.keys())} or None."
        )
    tf.keras.mixed_precision.set_global_policy(policy_name)
    logger.info("Mixed precision enabled: %s (%s)", mode, policy_name)


# ---------------------------------------------------------------------------
# Gradient accumulation helper
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """Accumulates gradients across multiple micro-batches.

    Useful for simulating large effective batch sizes on limited GPU memory.

    Args:
        model: Keras model whose variables gradients will be accumulated over.
        num_steps: Number of gradient accumulation steps (micro-batches).
    """

    def __init__(self, model: tf.keras.Model, num_steps: int = 1):
        self.model = model
        self.num_steps = num_steps
        self._accumulated_grads: Optional[List[tf.Variable]] = None
        self._step = 0

    def _init_accumulators(self) -> None:
        self._accumulated_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.model.trainable_variables
        ]

    def accumulate(self, grads: List[Optional[tf.Tensor]]) -> None:
        """Add a set of gradients to the accumulators.

        Args:
            grads: List of gradients (may contain None for unused params).
        """
        if self._accumulated_grads is None:
            self._init_accumulators()

        for acc, g in zip(self._accumulated_grads, grads):
            if g is not None:
                acc.assign_add(g / self.num_steps)

        self._step += 1

    def get_accumulated_gradients(self) -> List[tf.Tensor]:
        """Return the accumulated gradients as a list of tensors."""
        if self._accumulated_grads is None:
            return [tf.zeros_like(v) for v in self.model.trainable_variables]
        return [acc.read_value() for acc in self._accumulated_grads]

    def reset(self) -> None:
        """Zero out all gradient accumulators."""
        if self._accumulated_grads is not None:
            for acc in self._accumulated_grads:
                acc.assign(tf.zeros_like(acc))
        self._step = 0

    @property
    def ready(self) -> bool:
        """True when enough micro-batches have been accumulated."""
        return self._step >= self.num_steps


# ---------------------------------------------------------------------------
# DeepSpeed config parser
# ---------------------------------------------------------------------------

def load_deepspeed_config(config_path: str) -> Dict[str, Any]:
    """Load a DeepSpeed-format JSON configuration file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Dictionary with DeepSpeed settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info("Loaded DeepSpeed config from %s", config_path)
    return config


def apply_deepspeed_config(
    config: Dict[str, Any],
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Dict[str, Any]:
    """Apply relevant settings from a DeepSpeed config to TF model/optimizer.

    Currently supports:
    - ``optimizer.params.lr`` → sets optimizer learning rate
    - ``fp16.enabled`` / ``bf16.enabled`` → configures mixed precision

    Args:
        config: DeepSpeed config dictionary (from :func:`load_deepspeed_config`).
        model: The Keras model.
        optimizer: The Keras optimizer.

    Returns:
        Dict summarising applied settings.
    """
    applied: Dict[str, Any] = {}

    # Learning rate
    opt_cfg = config.get("optimizer", {}).get("params", {})
    lr = opt_cfg.get("lr")
    if lr is not None:
        optimizer.learning_rate.assign(float(lr))
        applied["learning_rate"] = lr

    # Mixed precision
    if config.get("fp16", {}).get("enabled"):
        configure_mixed_precision("fp16")
        applied["mixed_precision"] = "fp16"
    elif config.get("bf16", {}).get("enabled"):
        configure_mixed_precision("bf16")
        applied["mixed_precision"] = "bf16"

    logger.info("Applied DeepSpeed settings: %s", applied)
    return applied


# ---------------------------------------------------------------------------
# Distributed dataset helpers
# ---------------------------------------------------------------------------

def distribute_dataset(
    dataset: tf.data.Dataset,
    strategy: tf.distribute.Strategy,
) -> tf.distribute.DistributedDataset:
    """Wrap a tf.data.Dataset for use with a distribution strategy.

    Args:
        dataset: Input tf.data.Dataset.
        strategy: tf.distribute.Strategy to use.

    Returns:
        A DistributedDataset ready for use inside a strategy scope.
    """
    return strategy.experimental_distribute_dataset(dataset)
