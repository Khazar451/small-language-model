"""
Fine-tuning script for pre-trained Hugging Face models.

Usage:
    python scripts/finetune_pretrained.py --model_name gpt2 --task text_generation \\
        --train_data data/train.txt --output_dir outputs/finetuned_gpt2

    python scripts/finetune_pretrained.py --model_name bert-base-uncased \\
        --task sentiment_analysis --train_data data/train.csv --output_dir outputs/bert_sentiment
"""

import argparse
import logging
import os
import sys

import yaml
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.pretrained_wrapper import PretrainedModelWrapper
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained Hugging Face model."
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--task", type=str, default="text_generation",
        choices=["text_generation", "sequence_classification", "sentiment_analysis",
                 "question_answering"],
        help="Task type for fine-tuning.",
    )
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data file.")
    parser.add_argument("--val_data", type=str, help="Path to validation data file.")
    parser.add_argument("--output_dir", type=str, default="outputs/finetuned",
                        help="Directory to save fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of labels for classification tasks.")
    parser.add_argument("--freeze_base", action="store_true",
                        help="Freeze the base model weights.")
    parser.add_argument("--num_frozen_layers", type=int, default=0,
                        help="Number of base model layers to freeze.")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to training configuration YAML.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_text_for_generation(
    file_path: str, tokenizer, max_seq_length: int, batch_size: int
) -> tf.data.Dataset:
    """Load text data for language model fine-tuning."""
    from src.data.dataset import TextDataset
    dataset = TextDataset(file_path, tokenizer, max_seq_length=max_seq_length)
    return dataset.get_tf_dataset(batch_size=batch_size, shuffle=True)


def load_csv_for_classification(
    file_path: str, tokenizer, max_seq_length: int, batch_size: int,
    text_col: str = "text", label_col: str = "label", num_labels: int = 2
) -> tf.data.Dataset:
    """Load CSV data for classification fine-tuning."""
    import pandas as pd
    from src.data.dataset import ClassificationDataset

    df = pd.read_csv(file_path)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    dataset = ClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    return dataset.get_tf_dataset(batch_size=batch_size, shuffle=True)


def main():
    """Main fine-tuning entry point."""
    args = parse_args()
    tf.random.set_seed(args.seed)

    logger.info(
        "Fine-tuning '%s' for task '%s'", args.model_name, args.task
    )

    # Load model
    model = PretrainedModelWrapper(
        model_name=args.model_name,
        task=args.task,
        num_labels=args.num_labels,
        freeze_base=args.freeze_base,
        num_frozen_layers=args.num_frozen_layers,
    )

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset based on task
    if args.task == "text_generation":
        train_ds = load_text_for_generation(
            args.train_data, tokenizer, args.max_seq_length, args.batch_size
        )
        val_ds = None
        if args.val_data:
            val_ds = load_text_for_generation(
                args.val_data, tokenizer, args.max_seq_length, args.batch_size
            )
    elif args.task in ("sequence_classification", "sentiment_analysis"):
        train_ds = load_csv_for_classification(
            args.train_data, tokenizer, args.max_seq_length, args.batch_size,
            num_labels=args.num_labels,
        )
        val_ds = None
        if args.val_data:
            val_ds = load_csv_for_classification(
                args.val_data, tokenizer, args.max_seq_length, args.batch_size,
                num_labels=args.num_labels,
            )
    else:
        logger.error("Unsupported task for this script: %s", args.task)
        sys.exit(1)

    # Build optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        save_steps=500,
        early_stopping_patience=3,
        metric_for_best_model="val_loss",
    )

    logger.info("Starting fine-tuning...")
    history = trainer.train()

    # Save final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    logger.info("Fine-tuning complete! Model saved to %s", args.output_dir)
    logger.info("Training history: %s", history)


if __name__ == "__main__":
    main()
