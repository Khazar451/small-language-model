"""
Sentiment Analysis example using the small language model.

Demonstrates how to fine-tune for sentiment classification and run inference.

Usage:
    python examples/sentiment_analysis_example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_REVIEWS = [
    ("This movie was absolutely fantastic! Great performances.", 1),
    ("Terrible film. Waste of time and money.", 0),
    ("An okay movie, nothing special but not bad either.", 0),
    ("I loved every minute of this masterpiece!", 1),
    ("The plot was confusing and the acting was poor.", 0),
    ("One of the best films I have seen this year!", 1),
]

LABEL_MAP = {0: "negative", 1: "positive"}


def example_sentiment_dataset():
    """Demonstrate creating a sentiment analysis dataset."""
    print("=== Sentiment Analysis Dataset Creation ===\n")

    from transformers import AutoTokenizer
    from src.data.dataset import ClassificationDataset

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [review for review, _ in SAMPLE_REVIEWS]
    labels = [label for _, label in SAMPLE_REVIEWS]

    dataset = ClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_seq_length=128,
    )
    print(f"Created dataset with {len(dataset)} examples\n")

    tf_ds = dataset.get_tf_dataset(batch_size=2)
    for batch_inputs, batch_labels in tf_ds.take(1):
        print(f"Input IDs shape: {batch_inputs['input_ids'].shape}")
        print(f"Attention mask shape: {batch_inputs['attention_mask'].shape}")
        print(f"Labels: {batch_labels.numpy()}\n")


def example_sentiment_inference():
    """Demonstrate sentiment analysis inference."""
    print("=== Sentiment Analysis Inference ===\n")

    from transformers import AutoTokenizer
    from src.model.transformer import SmallTransformer, TransformerConfig
    from src.inference.predictor import Predictor

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_length=128,
        task="sentiment_analysis",
        num_labels=2,
    )
    model = SmallTransformer(config)

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        task="sentiment_analysis",
        max_seq_length=128,
    )

    texts = [review for review, _ in SAMPLE_REVIEWS]
    results = predictor.classify(texts, label_map=LABEL_MAP)

    print("Predictions (untrained model - random):")
    for text, result in zip(texts, results):
        label_name = result.get("label_name", result["label"])
        confidence = result["confidence"]
        print(f"  Text: {text[:60]}...")
        print(f"  Predicted: {label_name} ({confidence:.3f})\n")


def example_training_setup():
    """Show how to set up training for sentiment analysis."""
    print("=== Sentiment Analysis Training Setup ===\n")

    import tensorflow as tf
    from transformers import AutoTokenizer
    from src.model.transformer import SmallTransformer, TransformerConfig
    from src.data.dataset import ClassificationDataset
    from src.training.trainer import Trainer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [review for review, _ in SAMPLE_REVIEWS]
    labels = [label for _, label in SAMPLE_REVIEWS]

    dataset = ClassificationDataset(texts, labels, tokenizer, max_seq_length=128)
    tf_ds = dataset.get_tf_dataset(batch_size=2)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        task="sentiment_analysis",
        num_labels=2,
    )
    model = SmallTransformer(config)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=tf_ds,
        num_epochs=1,
        output_dir="/tmp/sentiment_checkpoints",
        logging_steps=1,
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print("Trainer configured. Call trainer.train() to start training.\n")
    print("(Skipping actual training in this demo)")


if __name__ == "__main__":
    print("Small Language Model - Sentiment Analysis Examples")
    print("=" * 50)
    example_sentiment_dataset()
    example_sentiment_inference()
    example_training_setup()
