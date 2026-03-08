"""
Quantization utilities for inference-time model compression.

Supports INT8 and simulated INT4 quantization for reducing model size and
memory footprint during inference.  These utilities are designed to work with
the SmallTransformer model and are compatible with TensorFlow Lite conversion.

Quantization methods:
- Dynamic INT8: Post-training dynamic range quantization (no calibration data)
- Static INT8: Post-training static quantization using representative dataset
- Simulated INT4: Weight-only INT4 quantization (simulated in FP32/FP16)
"""

import logging
import os
from typing import Optional, Callable, Iterator

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight quantization helpers
# ---------------------------------------------------------------------------

def quantize_weight_int8(weight: np.ndarray) -> tuple:
    """Quantize a weight array to INT8 using symmetric per-tensor quantization.

    Args:
        weight: NumPy float32 weight array.

    Returns:
        Tuple of (int8_weight, scale) where scale is a float32 scalar.
    """
    abs_max = np.max(np.abs(weight))
    if abs_max == 0:
        return weight.astype(np.int8), 1.0
    scale = abs_max / 127.0
    quantized = np.round(weight / scale).clip(-128, 127).astype(np.int8)
    return quantized, scale


def dequantize_weight_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 weights back to float32.

    Args:
        quantized: INT8 weight array.
        scale: Scale factor used during quantization.

    Returns:
        Dequantized float32 weight array.
    """
    return quantized.astype(np.float32) * scale


def quantize_weight_int4_simulated(weight: np.ndarray) -> tuple:
    """Simulate INT4 weight-only quantization using INT8 storage.

    INT4 values are in the range [-8, 7] and stored as INT8 for compatibility
    with standard numpy/TensorFlow operations.

    Args:
        weight: NumPy float32 weight array.

    Returns:
        Tuple of (int8_weight_with_int4_values, scale).
    """
    abs_max = np.max(np.abs(weight))
    if abs_max == 0:
        return weight.astype(np.int8), 1.0
    scale = abs_max / 7.0
    quantized = np.round(weight / scale).clip(-8, 7).astype(np.int8)
    return quantized, scale


def dequantize_weight_int4_simulated(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize simulated INT4 weights back to float32.

    Args:
        quantized: INT8 array containing INT4 values (range [-8, 7]).
        scale: Scale factor used during quantization.

    Returns:
        Dequantized float32 weight array.
    """
    return quantized.astype(np.float32) * scale


# ---------------------------------------------------------------------------
# Model-level quantization
# ---------------------------------------------------------------------------

class QuantizedWeightLayer(tf.keras.layers.Layer):
    """A wrapper that stores quantized weights and dequantizes on the fly.

    This enables weight-only quantization where weights are stored in a
    compressed format and expanded to float at runtime during the matmul.

    Args:
        original_layer: The Dense layer whose weights to quantize.
        mode: Quantization mode - 'int8' or 'int4'.
    """

    def __init__(
        self,
        units: int,
        quantized_kernel: np.ndarray,
        scale: float,
        mode: str = "int8",
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.q_scale = scale

        self._quantized_kernel = self.add_weight(
            name="quantized_kernel",
            shape=quantized_kernel.shape,
            dtype=tf.int8,
            initializer=tf.keras.initializers.Constant(quantized_kernel),
            trainable=False,
        )
        self._use_bias = use_bias

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Dequantize weights and perform the linear transformation."""
        kernel_fp32 = tf.cast(self._quantized_kernel, tf.float32) * self.q_scale
        return tf.linalg.matmul(x, kernel_fp32)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "mode": self.mode})
        return config


def quantize_model_weights(
    model: tf.keras.Model,
    mode: str = "int8",
) -> dict:
    """Extract and quantize all Dense layer weights in a model.

    Returns a dictionary mapping layer names to (quantized_kernel, scale) tuples
    that can be used for inspection or serialisation.

    Args:
        model: A built Keras model.
        mode: Quantization mode - 'int8' (default) or 'int4'.

    Returns:
        Dict mapping layer_name -> {'quantized': np.ndarray, 'scale': float,
        'shape': tuple, 'mode': str}
    """
    if mode not in ("int8", "int4"):
        raise ValueError(f"Unsupported quantization mode '{mode}'. Choose 'int8' or 'int4'.")

    quant_fn = quantize_weight_int8 if mode == "int8" else quantize_weight_int4_simulated
    results = {}

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel = layer.kernel.numpy()
            quantized, scale = quant_fn(kernel)
            results[layer.name] = {
                "quantized": quantized,
                "scale": scale,
                "shape": kernel.shape,
                "mode": mode,
            }
            logger.debug(
                "Quantized layer '%s': shape=%s, scale=%.6f, mode=%s",
                layer.name, kernel.shape, scale, mode,
            )

    logger.info(
        "Quantized %d Dense layers (mode=%s). "
        "Estimated size reduction: ~%.1fx",
        len(results),
        mode,
        4.0 if mode == "int4" else 2.0,
    )
    return results


def estimate_quantized_size_gb(model: tf.keras.Model, mode: str = "int8") -> float:
    """Estimate model size after quantization in gigabytes.

    Args:
        model: A built Keras model.
        mode: 'int8' (8-bit, 1 byte/param) or 'int4' (4-bit, 0.5 byte/param).

    Returns:
        Estimated size in GB.
    """
    total_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    bits_per_param = 8 if mode == "int8" else 4
    size_bytes = total_params * bits_per_param / 8
    return size_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# TFLite post-training quantization
# ---------------------------------------------------------------------------

def convert_to_tflite(
    model: tf.keras.Model,
    representative_dataset: Optional[Callable[[], Iterator]] = None,
    quantize: bool = True,
    output_path: Optional[str] = None,
) -> bytes:
    """Convert a Keras model to TensorFlow Lite with optional quantization.

    Args:
        model: A built and compiled Keras model.
        representative_dataset: Callable returning an iterator of sample inputs
            for calibration (required for full-integer quantization).
        quantize: Whether to apply post-training quantization.
        output_path: If given, save the .tflite file to this path.

    Returns:
        The serialised TFLite flatbuffer as bytes.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        logger.info("TFLite model saved to %s (%d bytes)", output_path, len(tflite_model))

    return tflite_model
