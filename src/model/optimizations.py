"""Memory and training optimization utilities for large transformer models.

This module provides helpers for:
- Gradient checkpointing (trade compute for memory during training)
- Mixed precision (float16) training
- Int8 quantization for inference memory reduction
- Automatic batch size selection based on available GPU memory
- Distributed training setup utilities

These optimizations are primarily intended for the 3B/5B parameter models
(``CONFIG_3B``, ``CONFIG_5B``) but can be used with any model size.

Example::

    from src.model.transformer import SmallTransformer, CONFIG_3B
    from src.model.optimizations import (
        enable_mixed_precision,
        apply_gradient_checkpointing,
        suggest_batch_size,
    )

    enable_mixed_precision()

    model = SmallTransformer(CONFIG_3B)
    apply_gradient_checkpointing(model)

    batch_size = suggest_batch_size(model_param_count=3e9, seq_len=2048)
    print(f"Recommended batch size: {batch_size}")
"""

import logging
import math
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BYTES_PER_MB = 1024 ** 2   # 1 MiB in bytes
_BYTES_PER_GB = 1024 ** 3   # 1 GiB in bytes

# Approximate number of bytes consumed per parameter on GPU:
#   fp32 training: weights(4) + gradients(4) + Adam m(4) + Adam v(4) = 16
#   fp16 training: fp16 weights(2) + fp32 master(4) + grads(2) + Adam(4+4) = ~8
#   fp16 inference: weights only at 2 bytes/param
_BYTES_PER_PARAM_TRAINING_FP32 = 16
_BYTES_PER_PARAM_TRAINING_FP16 = 8
_BYTES_PER_PARAM_INFERENCE = 2

# Approximate ratio used to estimate d_model from total parameter count.
# Derivation: params ≈ 12 * num_layers * d_model^2 for a GPT-style model
# (4*d^2 attention QKV/out projections + 2*d*4d FFN = 12*d^2 per layer).
# Using a representative num_layers factor of 12 gives the constant 144.
_PARAM_TO_D_MODEL_CONSTANT = 144  # 12 layers * 12 (per-layer d^2 coefficient)

# Approximate bytes of activation memory per token per dimension per layer.
# Each transformer block stores activations at each sub-layer residual, giving
# roughly 12 element-sized tensors of shape (seq_len, d_model).
_ACTIVATION_MULTIPLIER = 12


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def enable_mixed_precision(dtype: str = "float16") -> None:
    """Enable Keras mixed-precision globally.

    Sets the global Keras dtype policy to ``mixed_float16`` (or
    ``mixed_bfloat16``), which causes Dense and Embedding layers to use
    float16 compute while keeping weights in float32.  This roughly halves
    GPU memory for activations and speeds up matrix multiplications on
    modern GPUs.

    Should be called **before** constructing the model.

    Args:
        dtype: Base float type to mix. Supported values: ``"float16"``
            (recommended for NVIDIA Tensor Cores) and ``"bfloat16"``
            (recommended for TPUs and Ampere+ GPUs).

    Raises:
        ValueError: If *dtype* is not ``"float16"`` or ``"bfloat16"``.

    Example::

        from src.model.optimizations import enable_mixed_precision
        enable_mixed_precision()          # float16 default
        enable_mixed_precision("bfloat16")
    """
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(
            f"dtype must be 'float16' or 'bfloat16', got '{dtype}'"
        )
    policy_name = f"mixed_{dtype}"
    tf.keras.mixed_precision.set_global_policy(policy_name)
    logger.info("Mixed precision enabled: policy=%s", policy_name)


def disable_mixed_precision() -> None:
    """Reset the Keras dtype policy to the default float32."""
    tf.keras.mixed_precision.set_global_policy("float32")
    logger.info("Mixed precision disabled; policy reset to float32")


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def apply_gradient_checkpointing(model: tf.keras.Model) -> None:
    """Enable gradient checkpointing on a ``SmallTransformer`` model.

    Gradient checkpointing re-computes intermediate activations during the
    backward pass instead of storing them, which reduces peak memory usage
    at the cost of additional compute (~33% overhead).

    This function marks each ``TransformerBlock`` in ``model.blocks`` to use
    ``tf.recompute_grad`` so that their activations are freed after the
    forward pass and recomputed on demand during back-propagation.

    Args:
        model: A ``SmallTransformer`` instance.

    Note:
        This must be called **after** model construction but **before**
        compiling/training.

    Example::

        model = SmallTransformer(CONFIG_3B)
        apply_gradient_checkpointing(model)
    """
    if not hasattr(model, "blocks"):
        logger.warning(
            "apply_gradient_checkpointing: model has no 'blocks' attribute; "
            "skipping."
        )
        return

    for block in model.blocks:
        # Wrap the block's __call__ so TF recomputes activations on backward.
        original_call = block.__call__

        @tf.recompute_grad
        def _checkpointed(x, mask=None, training=False, _call=original_call):
            return _call(x, mask=mask, training=training)

        block.__call__ = _checkpointed

    logger.info(
        "Gradient checkpointing enabled for %d transformer blocks.",
        len(model.blocks),
    )


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model_for_inference(
    model: tf.keras.Model,
    quantization_type: str = "int8",
) -> tf.keras.Model:
    """Apply post-training quantization for inference memory reduction.

    Converts a trained ``SmallTransformer`` to a TensorFlow Lite (TFLite)
    int8 or float16 quantized model and returns a ``QuantizedModelWrapper``
    that exposes the same ``__call__`` interface.

    Args:
        model: A built (forward-passed) ``SmallTransformer`` instance.
        quantization_type: ``"int8"`` for integer quantization (smallest
            memory footprint) or ``"float16"`` for half-precision weights
            (higher accuracy than int8).

    Returns:
        A :class:`QuantizedModelWrapper` that runs the TFLite interpreter
        internally while presenting a Keras-like ``__call__`` interface.

    Raises:
        ValueError: If *quantization_type* is not ``"int8"`` or ``"float16"``.

    Example::

        model = SmallTransformer.load_pretrained("outputs/my_3b_model")
        quantized = quantize_model_for_inference(model, "float16")
        outputs = quantized(input_ids)
    """
    if quantization_type not in ("int8", "float16"):
        raise ValueError(
            f"quantization_type must be 'int8' or 'float16', "
            f"got '{quantization_type}'"
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization_type == "float16":
        converter.target_spec.supported_types = [tf.float16]
    else:
        # int8: dynamic range quantization (no representative dataset needed)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

    tflite_model = converter.convert()
    logger.info(
        "Model quantized to %s. TFLite model size: %.1f MB",
        quantization_type,
        len(tflite_model) / _BYTES_PER_MB,
    )
    return QuantizedModelWrapper(tflite_model)


class QuantizedModelWrapper:
    """Thin wrapper around a TFLite interpreter with a Keras-like interface.

    Args:
        tflite_model: Serialised TFLite flatbuffer (bytes) produced by
            :func:`quantize_model_for_inference`.
    """

    def __init__(self, tflite_model: bytes) -> None:
        self._interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def __call__(self, input_ids: tf.Tensor):
        """Run inference on *input_ids*.

        Args:
            input_ids: Integer tensor of shape ``(1, seq_len)``.
                Batch size must be 1 for TFLite fixed-shape inference.

        Returns:
            Dictionary ``{"logits": tf.Tensor}`` matching the
            ``SmallTransformer`` output format.
        """
        input_data = tf.cast(input_ids, tf.int32).numpy()
        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_data
        )
        self._interpreter.invoke()
        logits = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )
        return {"logits": tf.constant(logits)}


# ---------------------------------------------------------------------------
# Batch size suggestion
# ---------------------------------------------------------------------------

# Empirically-derived bytes of GPU memory consumed per parameter during
# training (fp32 weights + gradients + Adam moments ≈ 16 bytes/param).
_BYTES_PER_PARAM_TRAINING_FP32 = 16
_BYTES_PER_PARAM_TRAINING_FP16 = 8   # mixed precision cuts weights + grads
_BYTES_PER_PARAM_INFERENCE = 2        # float16 inference


def suggest_batch_size(
    model_param_count: float,
    seq_len: int,
    available_memory_gb: Optional[float] = None,
    mixed_precision: bool = False,
    for_inference: bool = False,
) -> int:
    """Suggest a per-GPU batch size based on model size and available memory.

    Uses a simple analytical model:

    .. code-block:: text

        memory_for_activations = batch_size * seq_len * d_model * bytes_per_elem
        total_memory = model_memory + activation_memory  <= available_memory * 0.85

    If *available_memory_gb* is ``None`` the function tries to query the first
    visible GPU via ``tf.config``; if no GPU is found it assumes 16 GB.

    Args:
        model_param_count: Total number of model parameters (e.g. ``3e9``).
        seq_len: Sequence length used during training/inference.
        available_memory_gb: GPU memory in GB. Auto-detected when ``None``.
        mixed_precision: Whether float16 mixed precision is enabled
            (halves model memory).
        for_inference: When ``True``, uses inference-only memory estimate
            (no gradients or optimizer states).

    Returns:
        Suggested batch size (at least 1).

    Example::

        batch = suggest_batch_size(3e9, seq_len=2048, mixed_precision=True)
        print(f"Use batch_size={batch}")
    """
    # --- determine available GPU memory ---
    if available_memory_gb is None:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # tf.config.experimental.get_memory_info returns bytes
                mem_info = tf.config.experimental.get_memory_info("GPU:0")
                available_memory_gb = mem_info["current"] / _BYTES_PER_GB
            except Exception:
                available_memory_gb = 16.0
        else:
            available_memory_gb = 16.0
        logger.debug(
            "suggest_batch_size: using %.1f GB available memory",
            available_memory_gb,
        )

    available_bytes = available_memory_gb * _BYTES_PER_GB * 0.85  # 15% safety margin

    # --- model static memory ---
    if for_inference:
        bytes_per_param = _BYTES_PER_PARAM_INFERENCE
    elif mixed_precision:
        bytes_per_param = _BYTES_PER_PARAM_TRAINING_FP16
    else:
        bytes_per_param = _BYTES_PER_PARAM_TRAINING_FP32

    model_bytes = model_param_count * bytes_per_param

    # --- activation memory per sample (rough estimate) ---
    # activations ≈ seq_len * d_model * _ACTIVATION_MULTIPLIER * bytes_per_elem
    # d_model is approximated from param count using:
    #   params ≈ _PARAM_TO_D_MODEL_CONSTANT * d_model^2
    bytes_per_elem = 2 if (mixed_precision or for_inference) else 4
    d_model_approx = max(256, int(math.sqrt(model_param_count / _PARAM_TO_D_MODEL_CONSTANT)))
    activation_bytes_per_sample = (
        seq_len * d_model_approx * bytes_per_elem * _ACTIVATION_MULTIPLIER
    )

    remaining = available_bytes - model_bytes
    if remaining <= 0:
        logger.warning(
            "suggest_batch_size: model alone (%.1f GB) exceeds available "
            "memory (%.1f GB). Consider mixed precision or model parallelism.",
            model_bytes / _BYTES_PER_GB,
            available_memory_gb,
        )
        return 1

    batch_size = max(1, int(remaining // activation_bytes_per_sample))
    # Round down to nearest power of 2 for efficiency
    batch_size = max(1, 2 ** int(math.log2(batch_size)))
    logger.info(
        "suggest_batch_size: model=%.1fGB available=%.1fGB -> batch_size=%d",
        model_bytes / _BYTES_PER_GB,
        available_memory_gb,
        batch_size,
    )
    return batch_size


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------

def create_distribution_strategy(
    strategy_type: str = "mirrored",
    num_gpus: Optional[int] = None,
    tpu_address: Optional[str] = None,
) -> tf.distribute.Strategy:
    """Create a TensorFlow distribution strategy for multi-GPU/TPU training.

    Args:
        strategy_type: One of:

            - ``"mirrored"`` – synchronous data-parallel training across
              multiple GPUs on a **single machine** (recommended for 3B models).
            - ``"multi_worker_mirrored"`` – synchronous data-parallel training
              across **multiple machines**, each with one or more GPUs
              (recommended for 5B models).
            - ``"tpu"`` – training on Google Cloud TPUs.
            - ``"default"`` – no-op single-device strategy.

        num_gpus: Number of GPUs to use for ``"mirrored"`` strategy.
            Defaults to all available GPUs when ``None``.
        tpu_address: TPU resolver address (e.g. ``"grpc://…"``).
            Required when *strategy_type* is ``"tpu"``.

    Returns:
        A ``tf.distribute.Strategy`` instance ready to use as a context
        manager around model construction and compilation.

    Raises:
        ValueError: If *strategy_type* is unrecognised.

    Example::

        strategy = create_distribution_strategy("mirrored")
        with strategy.scope():
            model = SmallTransformer(CONFIG_3B)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
            model.compile(optimizer=optimizer)
    """
    if strategy_type == "mirrored":
        gpus = tf.config.list_physical_devices("GPU")
        devices = None
        if num_gpus is not None:
            devices = [f"GPU:{i}" for i in range(min(num_gpus, len(gpus)))]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        logger.info(
            "MirroredStrategy created with %d replica(s).",
            strategy.num_replicas_in_sync,
        )
    elif strategy_type == "multi_worker_mirrored":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logger.info("MultiWorkerMirroredStrategy created.")
    elif strategy_type == "tpu":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address or ""
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info("TPUStrategy created.")
    elif strategy_type == "default":
        strategy = tf.distribute.get_strategy()
        logger.info("Using default (single-device) strategy.")
    else:
        raise ValueError(
            f"Unknown strategy_type '{strategy_type}'. "
            "Choose from: 'mirrored', 'multi_worker_mirrored', 'tpu', 'default'."
        )
    return strategy
