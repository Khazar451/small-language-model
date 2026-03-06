"""
Evaluation script for language models.

Usage:
    python scripts/evaluate.py --model_path outputs/checkpoints/best_model \\
        --test_data data/test.txt --task text_generation

    python scripts/evaluate.py --model_path outputs/finetuned/final_model \\
        --test_data data/test.csv --task sentiment_analysis
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained language model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory.")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data file.")
    parser.add_argument("--task", type=str, default="text_generation",
                        choices=["text_generation", "sequence_classification",
                                 "sentiment_analysis", "question_answering"],
                        help="Task type.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Evaluation batch size.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--output_file", type=str,
                        help="Path to save evaluation results.")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Load as HuggingFace pre-trained model.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of classification labels.")
    return parser.parse_args()


def evaluate_text_generation(model, tokenizer, test_path, batch_size, max_seq_length):
    """Evaluate language model on text generation task (perplexity)."""
    from src.data.dataset import TextDataset

    dataset = TextDataset(test_path, tokenizer, max_seq_length=max_seq_length)
    test_ds = dataset.get_tf_dataset(batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0

    for batch in test_ds:
        if isinstance(batch, (tuple, list)):
            input_ids, attention_mask = batch[0], batch[1]
        else:
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")

        outputs = model.compute_loss(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
            training=False,
        )
        total_loss += float(outputs["loss"])
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = float(np.exp(min(avg_loss, 20)))

    return {"loss": avg_loss, "perplexity": perplexity}


def evaluate_classification(model, tokenizer, test_path, batch_size, max_seq_length):
    """Evaluate classification model on accuracy, F1, and loss."""
    import pandas as pd
    from src.data.dataset import ClassificationDataset
    from src.training.metrics import compute_accuracy, compute_f1

    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    dataset = ClassificationDataset(
        texts=texts, labels=labels, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    test_ds = dataset.get_tf_dataset(batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0

    for batch_inputs, batch_labels in test_ds:
        outputs = model.compute_loss(
            input_ids=batch_inputs["input_ids"],
            labels=batch_labels,
            attention_mask=batch_inputs.get("attention_mask"),
            training=False,
        )
        if "loss" in outputs:
            total_loss += float(outputs["loss"])

        logits = outputs.get("logits")
        if logits is not None:
            preds = np.argmax(logits.numpy(), axis=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.numpy().tolist())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = compute_accuracy(all_preds, all_labels)
    f1 = compute_f1(all_preds, all_labels)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "num_samples": len(all_labels),
    }


def main():
    """Main evaluation entry point."""
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_pretrained:
        from src.model.pretrained_wrapper import PretrainedModelWrapper
        model = PretrainedModelWrapper.load_finetuned(
            args.model_path, task=args.task, num_labels=args.num_labels
        )
    else:
        from src.model.transformer import SmallTransformer
        model = SmallTransformer.load_pretrained(args.model_path)

    logger.info("Evaluating on %s", args.test_data)

    if args.task == "text_generation":
        results = evaluate_text_generation(
            model, tokenizer, args.test_data, args.batch_size, args.max_seq_length
        )
    elif args.task in ("sequence_classification", "sentiment_analysis"):
        results = evaluate_classification(
            model, tokenizer, args.test_data, args.batch_size, args.max_seq_length
        )
    else:
        logger.error("Evaluation for task '%s' not yet implemented.", args.task)
        sys.exit(1)

    logger.info("Evaluation results: %s", results)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
