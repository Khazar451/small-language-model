"""
Training script for the small language model from scratch.

Usage:
    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --train_data data/train.txt --num_epochs 3
"""

import argparse
import logging
import os
import sys

import yaml
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a small language model from scratch."
    )
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument("--train_data", type=str, help="Path to training text file.")
    parser.add_argument("--val_data", type=str, help="Path to validation text file.")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                        help="Path to model configuration YAML file.")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="HuggingFace tokenizer name.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Load and merge configuration from YAML and command-line args."""
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f).get("training", {})

    # Command-line overrides
    if args.train_data:
        config["train_data_path"] = args.train_data
    if args.val_data:
        config["val_data_path"] = args.val_data
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate

    return config


def main():
    """Main training entry point."""
    args = parse_args()

    # Set random seeds
    tf.random.set_seed(args.seed)

    # Load configurations
    train_cfg = load_config(args)

    model_cfg = {}
    if os.path.exists(args.model_config):
        with open(args.model_config, "r") as f:
            model_cfg = yaml.safe_load(f).get("model", {})

    logger.info("Building model...")
    config = TransformerConfig.from_dict(model_cfg)
    model = SmallTransformer(config)

    # Load tokenizer and datasets
    from transformers import AutoTokenizer
    tokenizer_name = train_cfg.get("tokenizer_name", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from src.data.dataset import TextDataset

    train_path = train_cfg.get("train_data_path", "data/train.txt")
    val_path = train_cfg.get("val_data_path", "data/val.txt")

    if not os.path.exists(train_path):
        logger.error("Training data not found at %s", train_path)
        sys.exit(1)

    max_seq_length = train_cfg.get("max_seq_length", args.max_seq_length)
    batch_size = train_cfg.get("batch_size", 8)

    logger.info("Loading training dataset from %s", train_path)
    train_dataset_obj = TextDataset(train_path, tokenizer, max_seq_length=max_seq_length)
    train_ds = train_dataset_obj.get_tf_dataset(batch_size=batch_size, shuffle=True)

    val_ds = None
    if os.path.exists(val_path):
        logger.info("Loading validation dataset from %s", val_path)
        val_dataset_obj = TextDataset(val_path, tokenizer, max_seq_length=max_seq_length)
        val_ds = val_dataset_obj.get_tf_dataset(batch_size=batch_size, shuffle=False)

    # Build optimizer with learning rate schedule
    lr = train_cfg.get("learning_rate", 5e-5)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    output_dir = train_cfg.get("output_dir", args.output_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=train_cfg.get("num_epochs", 3),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        output_dir=output_dir,
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg.get("logging_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 0),
        early_stopping_patience=(
            train_cfg.get("early_stopping_patience", 3)
            if train_cfg.get("early_stopping", False) else 0
        ),
    )

    logger.info("Starting training...")
    history = trainer.train()
    logger.info("Training complete. History: %s", history)

    # Save tokenizer alongside the model so inference can reload it
    tokenizer.save_pretrained(output_dir)
    logger.info("Tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    main()
