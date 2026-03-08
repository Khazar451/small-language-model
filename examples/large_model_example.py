"""
Example: Training and running inference with 3B / 5B parameter models.

This script demonstrates how to:
1. Choose between the predefined 3B and 5B configurations.
2. Enable mixed precision and gradient checkpointing before building the model.
3. Wrap the model in a multi-GPU distribution strategy.
4. Use suggest_batch_size() to pick a per-GPU batch size automatically.
5. Run a minimal forward pass (no real data required).

Prerequisites
-------------
    pip install tensorflow numpy

For 3B training you need at least 2× 40 GB GPUs.
For 5B training you need at least 4× 40 GB GPUs (or 2× 80 GB).

Usage
-----
    # 3B model (default)
    python examples/large_model_example.py

    # 5B model
    python examples/large_model_example.py --model 5b

    # Show estimated parameters without building the model
    python examples/large_model_example.py --dry-run
"""

import argparse
import logging
import sys
import os

# Allow running directly from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from src.model.transformer import SmallTransformer, CONFIG_3B, CONFIG_5B
from src.model.optimizations import (
    enable_mixed_precision,
    apply_gradient_checkpointing,
    create_distribution_strategy,
    suggest_batch_size,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def estimate_params(config) -> int:
    """Estimate total parameters analytically (no model build required)."""
    d = config.d_model
    v = config.vocab_size
    L = config.num_layers
    d_ff = config.d_ff
    s = config.max_seq_length

    # Embeddings: token + position
    emb = v * d + s * d
    # Each transformer block:
    #   - 4 attention projection matrices (Q, K, V, O): 4 * d^2
    #   - 2 FFN weight matrices (up-project d→d_ff, down-project d_ff→d): 2 * d * d_ff
    block = L * (4 * d * d + 2 * d * d_ff)
    # LM head (tied weights not counted, but head bias negligible)
    total = emb + block
    return total


def main():
    parser = argparse.ArgumentParser(description="Large model (3B/5B) example")
    parser.add_argument(
        "--model",
        choices=["3b", "5b"],
        default="3b",
        help="Which predefined config to use (default: 3b)",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "mirrored"],
        default="default",
        help="Distribution strategy (default: default / single GPU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print parameter estimate; skip model build",
    )
    args = parser.parse_args()

    config = CONFIG_3B if args.model == "3b" else CONFIG_5B
    label = "3B" if args.model == "3b" else "5B"

    estimated = estimate_params(config)
    logger.info(
        "=== %s model configuration ===\n"
        "  d_model=%d  num_layers=%d  num_heads=%d  d_ff=%d\n"
        "  max_seq_length=%d  vocab_size=%d\n"
        "  gradient_checkpointing=%s  use_mixed_precision=%s\n"
        "  Estimated parameters: %s",
        label,
        config.d_model, config.num_layers, config.num_heads, config.d_ff,
        config.max_seq_length, config.vocab_size,
        config.gradient_checkpointing, config.use_mixed_precision,
        f"{estimated:,}",
    )

    if args.dry_run:
        logger.info("--dry-run: skipping model construction.")
        return

    # 1. Mixed precision (must be set before model construction)
    if config.use_mixed_precision:
        logger.info("Enabling float16 mixed precision …")
        enable_mixed_precision("float16")

    # 2. Distribution strategy
    strategy = create_distribution_strategy(args.strategy)

    # 3. Build model inside the strategy scope
    logger.info("Building %s model …", label)
    with strategy.scope():
        model = SmallTransformer(config)

    # 4. Gradient checkpointing (reduces peak memory ~40%)
    if config.gradient_checkpointing:
        logger.info("Applying gradient checkpointing …")
        apply_gradient_checkpointing(model)

    # 5. Suggest per-GPU batch size
    recommended_batch = suggest_batch_size(
        model_param_count=float(estimated),
        seq_len=config.max_seq_length,
        mixed_precision=config.use_mixed_precision,
    )
    logger.info("Recommended per-GPU batch size: %d", recommended_batch)

    # 6. Minimal forward pass with random data (tiny seq for speed)
    test_seq_len = 16
    dummy_input = tf.constant(
        np.random.randint(1, config.vocab_size, (1, test_seq_len)), dtype=tf.int32
    )
    logger.info("Running forward pass (seq_len=%d) …", test_seq_len)
    outputs = model(dummy_input, training=False)
    logits_shape = tuple(outputs["logits"].shape)
    logger.info("Output logits shape: %s  ✓", logits_shape)

    # 7. Build model to get actual parameter count
    actual = model.count_parameters()
    logger.info("Actual trainable parameters: %s", f"{actual:,}")

    logger.info(
        "\nTo train this model, use:\n"
        "  strategy = create_distribution_strategy('mirrored', num_gpus=N)\n"
        "  with strategy.scope():\n"
        "      model = SmallTransformer(config)\n"
        "  apply_gradient_checkpointing(model)\n"
        "  # … compile and fit with batch_size=%d\n",
        recommended_batch,
    )


if __name__ == "__main__":
    main()
