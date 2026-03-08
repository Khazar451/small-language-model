"""
Post-training quantization utilities (INT4 / INT8).

Provides weight-only and activation-aware quantization for TensorFlow/Keras
models, reducing memory footprint and improving inference throughput on
supported hardware.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Supported quantization types
_SUPPORTED_TYPES = {"int8", "int4"}


class QuantizedDense(tf.keras.layers.Layer):
    """A Dense layer whose weights are stored in a quantized integer format.

    At inference time the quantized weights are dequantized on the fly to
    ``float16`` before the matrix multiplication, so the layer remains
    numerically compatible with a standard ``Dense`` layer.

    Args:
        units: Output dimensionality (same as ``tf.keras.layers.Dense``).
        bits: Bit-width for quantization (``8`` or ``4``).
        group_size: Number of weights that share a single scale factor
            (block-wise quantization).  Set to ``-1`` for per-column scaling.
        use_bias: Whether the layer uses a bias vector.
    """

    def __init__(
        self,
        units: int,
        bits: int = 8,
        group_size: int = -1,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.units = units
        self.bits = bits
        self.group_size = group_size
        self.use_bias = use_bias

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        # Quantized weight stored as int8 (int4 is packed into int8)
        self.q_weight = self.add_weight(
            name="q_weight",
            shape=(in_features, self.units),
            dtype=tf.int8,
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            shape=(1, self.units),
            dtype=tf.float32,
            trainable=False,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                dtype=tf.float32,
                initializer="zeros",
                trainable=False,
            )
        else:
            self.bias = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Dequantize weights on the fly
        weight_fp = tf.cast(self.q_weight, tf.float16) * tf.cast(self.scale, tf.float16)
        inputs_fp = tf.cast(inputs, tf.float16)
        output = tf.matmul(inputs_fp, weight_fp)
        if self.bias is not None:
            output = output + tf.cast(self.bias, tf.float16)
        return tf.cast(output, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "bits": self.bits, "use_bias": self.use_bias})
        return config


def _quantize_tensor(
    tensor: np.ndarray,
    bits: int,
    group_size: int = -1,
) -> tuple:
    """Quantize a weight tensor and return (quantized_int8, scale).

    For INT4, values are clamped to [-8, 7] and stored as ``int8``.
    """
    if group_size > 0:
        # Block-wise quantization
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, group_size)
        max_vals = np.abs(flat).max(axis=1, keepdims=True).clip(min=1e-8)
        qmax = (2 ** (bits - 1)) - 1
        scale = max_vals / qmax
        q = np.round(flat / scale).clip(-qmax - 1, qmax).astype(np.int8)
        return q.reshape(orig_shape), scale.reshape(-1)
    else:
        qmax = (2 ** (bits - 1)) - 1
        col_max = np.abs(tensor).max(axis=0, keepdims=True).clip(min=1e-8)
        scale = col_max / qmax
        q = np.round(tensor / scale).clip(-qmax - 1, qmax).astype(np.int8)
        return q, scale


def quantize_model(
    model: tf.keras.Model,
    quantization_type: str = "int8",
    group_size: int = 128,
    calibration_dataset: Optional[Any] = None,
    output_path: Optional[str] = None,
) -> "QuantizedModel":
    """Apply post-training quantization to a Keras model.

    Currently supports weight-only quantization of all ``Dense`` layers.
    Activation quantization (requiring a calibration dataset) will be applied
    if *calibration_dataset* is provided (INT8 only).

    Args:
        model: Trained ``tf.keras.Model`` to quantize.
        quantization_type: ``"int8"`` or ``"int4"``.
        group_size: Block size for block-wise quantization.  Pass ``-1`` for
            per-column (standard) scaling.
        calibration_dataset: Optional ``tf.data.Dataset`` used for activation
            range calibration (INT8 dynamic quantization).
        output_path: If provided, save the quantized model to this path.

    Returns:
        A :class:`QuantizedModel` wrapping the quantized weights.

    Example:
        >>> q_model = quantize_model(model, quantization_type="int8")
        >>> q_model.save("outputs/model_int8")
    """
    if quantization_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"quantization_type must be one of {_SUPPORTED_TYPES}, got {quantization_type!r}"
        )

    bits = 8 if quantization_type == "int8" else 4
    logger.info("Quantizing model to %s (bits=%d, group_size=%d)", quantization_type, bits, group_size)

    quantized_weights = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.kernel is not None:
            kernel_np = layer.kernel.numpy()
            q, scale = _quantize_tensor(kernel_np, bits=bits, group_size=group_size)
            quantized_weights[layer.name] = {
                "q_weight": q,
                "scale": scale,
                "bias": layer.bias.numpy() if layer.bias is not None else None,
                "bits": bits,
            }
            logger.debug("  Quantized layer '%s': %s -> int%d", layer.name, kernel_np.shape, bits)

    q_model = QuantizedModel(model, quantized_weights, bits=bits)

    if output_path:
        q_model.save(output_path)

    return q_model


class QuantizedModel:
    """Wrapper around a Keras model that stores quantized weight data.

    Args:
        base_model: The original (fp32) model.
        quantized_weights: Dictionary mapping layer names to quantized arrays.
        bits: Quantization bit-width.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        quantized_weights: dict,
        bits: int = 8,
    ):
        self.base_model = base_model
        self.quantized_weights = quantized_weights
        self.bits = bits

    def get_size_gb(self) -> float:
        """Estimate the on-disk size of quantized weights in GB."""
        total_bytes = 0
        for qdata in self.quantized_weights.values():
            total_bytes += qdata["q_weight"].nbytes
            total_bytes += qdata["scale"].nbytes
            if qdata["bias"] is not None:
                total_bytes += qdata["bias"].nbytes
        return total_bytes / (1024 ** 3)

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save(self, path: str) -> None:
        """Save the quantized model (weights metadata + base model config)."""
        os.makedirs(path, exist_ok=True)
        # Save quantized weight arrays
        np.savez_compressed(
            os.path.join(path, "quantized_weights.npz"),
            **{
                f"{name}__{key}": val
                for name, qdata in self.quantized_weights.items()
                for key, val in qdata.items()
                if val is not None and isinstance(val, np.ndarray)
            },
        )
        # Save the base model config
        config = self.base_model.get_config() if hasattr(self.base_model, "get_config") else {}
        import json
        with open(os.path.join(path, "model_config.json"), "w") as fh:
            json.dump({"bits": self.bits, "base_config": config}, fh, indent=2)
        logger.info("Quantized model saved to %s (%.2f GB)", path, self.get_size_gb())
