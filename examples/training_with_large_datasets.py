"""
Example: Training with large datasets using streaming and multi-source mixing.

Run:
    python examples/training_with_large_datasets.py \\
        --config config/training_config.yaml \\
        --data_config config/data_config.yaml \\
        --output_dir outputs/large_run
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from transformers import AutoTokenizer

import tensorflow as tf
from src.data.huggingface_loader import HuggingFaceDataLoader
from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--data_config", default="config/data_config.yaml")
    parser.add_argument("--output_dir", default="outputs/large_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        train_cfg = yaml.safe_load(fh).get("training", {})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Build multi-source interleaved dataset (streaming)
    loader = HuggingFaceDataLoader(
        datasets=["wikitext", "openwebtext"],
        weights=[0.4, 0.6],
        streaming=True,
        dataset_configs={"wikitext": "wikitext-103-raw-v1"},
    )
    train_ds = loader.get_interleaved_dataset(
        tokenizer=tokenizer,
        batch_size=train_cfg.get("batch_size", 4),
        max_seq_length=2048,
    )

    # Build model with modern architecture optimizations
    config = TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_kv_heads=4,
        num_layers=24,
        d_ff=8192,
        positional_encoding="rope",
        activation="swiglu",
        use_flash_attention=True,
    )
    model = SmallTransformer(config)
    logger.info("Parameters: %d", model.count_parameters())

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        gradient_checkpointing=True,
        mixed_precision=True,
        checkpoint_dir=args.output_dir,
        resume_from_checkpoint=True,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        num_epochs=train_cfg.get("num_epochs", 3),
        logging_steps=50,
    )
    trainer.train()


if __name__ == "__main__":
    main()
