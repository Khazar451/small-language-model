"""
Training example for a 3B-parameter model.

This script demonstrates how to initialize and train a 3B-parameter
SmallTransformer with production-grade optimizations: GQA, RoPE, SwiGLU,
gradient checkpointing, and FP16 mixed precision.

Usage:
    python examples/train_3b_model.py

Requirements:
    - At least 8 GB GPU VRAM (FP16) or 16 GB CPU RAM
    - pip install tensorflow transformers
"""

import os
import sys
import logging

import numpy as np
import tensorflow as tf

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.trainer import Trainer
from src.training.distributed import configure_mixed_precision, auto_detect_strategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_3b_config(task: str = "text_generation") -> TransformerConfig:
    """Return the production 3B configuration with all optimizations."""
    return TransformerConfig(
        vocab_size=50257,
        d_model=1536,
        num_heads=16,
        num_kv_heads=4,       # GQA: 4 KV heads shared across 16 Q heads
        num_layers=24,
        d_ff=6144,
        max_seq_length=2048,
        dropout_rate=0.1,
        attention_dropout=0.1,
        task=task,
        use_gqa=True,          # Grouped-Query Attention
        use_rope=True,         # Rotary Position Embeddings
        use_swiglu=True,       # SwiGLU FFN
        use_flash_attention=False,  # Set True if Flash Attention kernel available
        gradient_checkpointing=True,  # ~30-40% training memory reduction
        mixed_precision="fp16",       # ~2x memory savings
    )


# ---------------------------------------------------------------------------
# Synthetic dataset for demonstration
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    vocab_size: int,
    seq_length: int,
    num_samples: int = 1000,
    batch_size: int = 2,
) -> tf.data.Dataset:
    """Create a random token dataset for demonstration/benchmarking."""
    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, vocab_size, size=(num_samples, seq_length)).astype(np.int32)
    attention_mask = np.ones((num_samples, seq_length), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Auto-detect best strategy (TPU > multi-GPU > single-GPU > CPU)
    strategy = auto_detect_strategy()

    # 2. Enable mixed precision globally (FP16)
    configure_mixed_precision("fp16")

    # 3. Build model inside the strategy scope
    with strategy.scope():
        config = build_3b_config()
        model = SmallTransformer(config)

        # Warm-up build with a small dummy input
        dummy = tf.zeros((1, 8), dtype=tf.int32)
        _ = model(dummy)

    param_count = model.count_parameters()
    logger.info(
        "3B model initialized: %d parameters (%.2fB)",
        param_count,
        param_count / 1e9,
    )

    # 4. Build synthetic datasets (replace with real data for actual training)
    seq_len = 512  # Use shorter sequences for quick demo; 2048 for real training
    train_ds = make_synthetic_dataset(config.vocab_size, seq_len, num_samples=64, batch_size=2)
    val_ds = make_synthetic_dataset(config.vocab_size, seq_len, num_samples=16, batch_size=2)

    # 5. Create Trainer
    output_dir = os.path.join("outputs", "3b_model")
    trainer = Trainer(
        model=model,
        optimizer="adamw",
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        output_dir=output_dir,
        save_steps=50,
        logging_steps=10,
        eval_steps=50,
    )

    # 6. Train
    logger.info("Starting training …")
    history = trainer.train()
    logger.info("Training complete. Final loss: %.4f", history["train_loss"][-1])

    # 7. Save
    save_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(save_path)
    logger.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
