"""
End-to-end script for training a 3B parameter language model.

Usage:
    python scripts/train_3b_model.py \\
        --config config/training_config.yaml \\
        --train_data data/processed/train \\
        --val_data data/processed/val \\
        --output_dir outputs/3b_model \\
        --num_gpus 4
"""

import argparse
import logging
import os
import sys

import yaml

# Allow running from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

import tensorflow as tf
from src.data.streaming_dataset import StreamingDataset
from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 3B model architecture (approximately 3 billion parameters)
_3B_CONFIG = {
    "vocab_size": 50257,
    "d_model": 2560,
    "num_heads": 20,
    "num_kv_heads": 5,
    "num_layers": 32,
    "d_ff": 10240,
    "max_seq_length": 2048,
    "dropout_rate": 0.0,
    "positional_encoding": "rope",
    "activation": "swiglu",
    "use_flash_attention": True,
    "task": "text_generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3B parameter language model")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--train_data", required=True, help="Glob pattern or directory for training shards")
    parser.add_argument("--val_data", default=None, help="Glob pattern or directory for validation shards")
    parser.add_argument("--output_dir", default="outputs/3b_model")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0 = all available)")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load training config
    with open(args.config) as fh:
        train_cfg = yaml.safe_load(fh).get("training", {})

    batch_size = args.batch_size or train_cfg.get("batch_size", 2)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 3)

    # Set up GPU strategy
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Enable memory growth to avoid allocating all GPU memory upfront
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        num_gpus = args.num_gpus or len(gpus)
        strategy = tf.distribute.MirroredStrategy(
            devices=[gpus[i].name for i in range(min(num_gpus, len(gpus)))]
        )
        logger.info("Using MirroredStrategy with %d GPU(s)", strategy.num_replicas_in_sync)
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        logger.warning("No GPUs found; training on CPU (very slow for 3B models)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # Build datasets
    train_sources = (
        [os.path.join(args.train_data, "*.jsonl"), os.path.join(args.train_data, "*.txt")]
        if os.path.isdir(args.train_data)
        else [args.train_data]
    )
    train_ds = StreamingDataset(
        sources=train_sources,
        tokenizer=tokenizer,
        max_seq_length=_3B_CONFIG["max_seq_length"],
    ).get_tf_dataset(batch_size=batch_size, shuffle=True)

    val_ds = None
    if args.val_data:
        val_sources = (
            [os.path.join(args.val_data, "*.jsonl"), os.path.join(args.val_data, "*.txt")]
            if os.path.isdir(args.val_data)
            else [args.val_data]
        )
        val_ds = StreamingDataset(
            sources=val_sources,
            tokenizer=tokenizer,
            max_seq_length=_3B_CONFIG["max_seq_length"],
        ).get_tf_dataset(batch_size=batch_size, shuffle=False)

    with strategy.scope():
        config = TransformerConfig(**_3B_CONFIG)
        model = SmallTransformer(config)
        logger.info("Model parameters: %d", model.count_parameters())

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            beta_1=0.9,
            beta_2=0.95,
            epsilon=1e-8,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=num_epochs,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        checkpoint_dir=args.output_dir,
        save_steps=train_cfg.get("save_steps", 1000),
        logging_steps=train_cfg.get("logging_steps", 50),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        mixed_precision=train_cfg.get("mixed_precision", "bf16") not in (None, False, ""),
        resume_from_checkpoint=args.resume or train_cfg.get("resume_from_checkpoint", False),
    )
    trainer.train()


if __name__ == "__main__":
    main()
