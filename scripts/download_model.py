"""
Script to download pre-trained models from HuggingFace.

Usage:
    python scripts/download_model.py --model_name gpt2 --output_dir models/gpt2
    python scripts/download_model.py --model_name bert-base-uncased
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "distilgpt2",
    "roberta-base",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a pre-trained model from HuggingFace Hub."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help=f"Model name. Supported: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Directory to save the model. Defaults to 'models/<model_name>'.",
    )
    parser.add_argument(
        "--task", type=str, default="text_generation",
        choices=["text_generation", "sequence_classification",
                 "sentiment_analysis", "question_answering"],
        help="Task type (affects which model head is downloaded).",
    )
    parser.add_argument(
        "--list_models", action="store_true",
        help="List all supported models and exit.",
    )
    return parser.parse_args()


def main():
    """Main download entry point."""
    args = parse_args()

    if args.list_models:
        print("Supported pre-trained models:")
        for model in SUPPORTED_MODELS:
            print(f"  - {model}")
        return

    output_dir = args.output_dir or os.path.join("models", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Downloading model '%s' for task '%s'...", args.model_name, args.task)

    try:
        from transformers import AutoTokenizer

        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer saved to %s", output_dir)

        # Download model
        logger.info("Downloading model weights...")
        if args.task == "text_generation":
            from transformers import TFAutoModelForCausalLM
            model = TFAutoModelForCausalLM.from_pretrained(args.model_name)
        elif args.task in ("sequence_classification", "sentiment_analysis"):
            from transformers import TFAutoModelForSequenceClassification
            model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
        elif args.task == "question_answering":
            from transformers import TFAutoModelForQuestionAnswering
            model = TFAutoModelForQuestionAnswering.from_pretrained(args.model_name)
        else:
            from transformers import TFAutoModel
            model = TFAutoModel.from_pretrained(args.model_name)

        model.save_pretrained(output_dir)
        logger.info("Model saved to %s", output_dir)

        # Print model info
        total_params = sum(p.numpy().size for p in model.weights)
        logger.info(
            "Download complete! Model: %s | Parameters: %s | Saved to: %s",
            args.model_name,
            f"{total_params:,}",
            output_dir,
        )

    except ImportError as exc:
        logger.error("transformers is required: pip install transformers")
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to download model '%s': %s", args.model_name, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
