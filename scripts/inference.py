"""
Inference script for running predictions with a trained model.

Usage:
    # Text generation
    python scripts/inference.py --model_path outputs/finetuned_gpt2 \\
        --task text_generation --prompt "Once upon a time"

    # Sentiment analysis
    python scripts/inference.py --model_path outputs/bert_sentiment \\
        --task sentiment_analysis --input_file data/texts.txt

    # Q&A
    python scripts/inference.py --model_path outputs/qa_model \\
        --task question_answering \\
        --question "What is the capital of France?" \\
        --context "Paris is the capital of France."
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained language model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory.")
    parser.add_argument("--task", type=str, default="text_generation",
                        choices=["text_generation", "sequence_classification",
                                 "sentiment_analysis", "question_answering"],
                        help="Inference task.")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation.")
    parser.add_argument("--input_file", type=str,
                        help="File with one input text per line.")
    parser.add_argument("--output_file", type=str,
                        help="File to save predictions.")
    parser.add_argument("--question", type=str, help="Question for Q&A task.")
    parser.add_argument("--context", type=str, help="Context passage for Q&A task.")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum new tokens for text generation.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference.")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Load as HuggingFace pre-trained model.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of classification labels.")
    parser.add_argument("--label_map", type=str,
                        help="JSON string mapping label indices to names "
                             "(e.g., '{\"0\": \"negative\", \"1\": \"positive\"}').")
    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)

    from src.inference.predictor import Predictor

    predictor = Predictor.from_pretrained(
        model_path=args.model_path,
        task=args.task,
        num_labels=args.num_labels,
    )

    label_map = None
    if args.label_map:
        label_map = {int(k): v for k, v in json.loads(args.label_map).items()}

    results = []

    if args.task == "text_generation":
        prompts = []
        if args.prompt:
            prompts.append(args.prompt)
        if args.input_file:
            with open(args.input_file) as f:
                prompts.extend(line.strip() for line in f if line.strip())

        if not prompts:
            logger.error("Provide --prompt or --input_file for text generation.")
            sys.exit(1)

        for prompt in prompts:
            result = predictor.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            results.append(result)
            print(f"\nPrompt: {prompt}")
            for text in result["generated_text"]:
                print(f"Generated: {text}")

    elif args.task in ("sequence_classification", "sentiment_analysis"):
        texts = []
        if args.input_file:
            with open(args.input_file) as f:
                texts = [line.strip() for line in f if line.strip()]
        elif args.prompt:
            texts = [args.prompt]
        else:
            logger.error("Provide --input_file or --prompt for classification.")
            sys.exit(1)

        results = predictor.classify(texts, label_map=label_map, batch_size=args.batch_size)

        for text, result in zip(texts, results):
            label_name = result.get("label_name", result["label"])
            confidence = result["confidence"]
            print(f"Text: {text[:80]}...")
            print(f"  Label: {label_name} (confidence: {confidence:.3f})")

    elif args.task == "question_answering":
        if not args.question or not args.context:
            logger.error("Provide --question and --context for Q&A.")
            sys.exit(1)

        result = predictor.answer_question(
            question=args.question,
            context=args.context,
        )
        results.append(result)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']:.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
