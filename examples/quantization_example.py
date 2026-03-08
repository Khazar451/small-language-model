"""
Example: INT4/INT8 post-training quantization.

Run:
    python examples/quantization_example.py \\
        --model_path outputs/my_model \\
        --quantization int4 \\
        --output_path outputs/my_model_int4
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.model.quantization import quantize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--quantization", default="int4", choices=["int8", "int4"])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)
    model = tf.keras.models.load_model(args.model_path)

    fp32_params = sum(w.numpy().nbytes for w in model.trainable_weights)
    logger.info("FP32 model size: %.2f GB", fp32_params / 1024 ** 3)

    # Apply quantization
    q_model = quantize_model(
        model,
        quantization_type=args.quantization,
        group_size=args.group_size,
    )

    logger.info(
        "%s model size: %.2f GB  (%.1f%% of FP32)",
        args.quantization.upper(),
        q_model.get_size_gb(),
        q_model.get_size_gb() / (fp32_params / 1024 ** 3) * 100,
    )

    q_model.save(args.output_path)
    logger.info("Quantized model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
