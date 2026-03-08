"""
Post-training quantization script.

Usage:
    python scripts/quantize_model.py \\
        --model_path outputs/my_model \\
        --quantization int8 \\
        --output_path outputs/my_model_int8

    python scripts/quantize_model.py \\
        --model_path outputs/my_model \\
        --quantization int4 \\
        --group_size 128 \\
        --output_path outputs/my_model_int4
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.model.quantization import quantize_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply post-training quantization to a saved model")
    parser.add_argument("--model_path", required=True, help="Path to the saved Keras model")
    parser.add_argument("--quantization", default="int8", choices=["int8", "int4"],
                        help="Quantization type (default: int8)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Block size for block-wise quantization (-1 for per-column)")
    parser.add_argument("--output_path", required=True, help="Path to save the quantized model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error("Model path not found: %s", args.model_path)
        sys.exit(1)

    logger.info("Loading model from %s", args.model_path)
    model = tf.keras.models.load_model(args.model_path)

    logger.info("Applying %s quantization (group_size=%d)", args.quantization, args.group_size)
    q_model = quantize_model(
        model,
        quantization_type=args.quantization,
        group_size=args.group_size,
    )

    logger.info("Quantized model size: %.2f GB", q_model.get_size_gb())
    q_model.save(args.output_path)
    logger.info("Quantized model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
