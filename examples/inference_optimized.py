"""
Example: Optimized inference with INT8 quantization and KV-cache.

Run:
    python examples/inference_optimized.py \\
        --model_path outputs/my_model \\
        --prompts "The future of AI is" "Large language models can"
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

import tensorflow as tf
from src.model.quantization import quantize_model
from src.inference.optimized_inference import OptimizedPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the saved model")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--quantize", default="int8", choices=["none", "int8", "int4"],
                        help="Optional quantization before inference (default: int8)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompts", nargs="+",
                        default=["The future of AI is", "Large language models can"])
    parser.add_argument("--output_file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model from %s", args.model_path)
    model = tf.keras.models.load_model(args.model_path)

    # Optionally quantize
    if args.quantize != "none":
        logger.info("Quantizing model to %s", args.quantize)
        q_model = quantize_model(model, quantization_type=args.quantize)
        logger.info("Quantized model size: %.2f GB", q_model.get_size_gb())
        inference_model = q_model.base_model
    else:
        inference_model = model

    # Build optimized predictor
    predictor = OptimizedPredictor(
        model=inference_model,
        tokenizer=tokenizer,
        use_kv_cache=True,
        dtype="float16",
    )

    # Generate
    logger.info("Generating %d prompts…", len(args.prompts))
    outputs = predictor.generate_batch(
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for i, (prompt, text) in enumerate(zip(args.prompts, outputs)):
        print(f"\n--- Prompt {i + 1} ---")
        print(f"Input : {prompt}")
        print(f"Output: {text}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as fh:
            for text in outputs:
                fh.write(text + "\n---\n")
        logger.info("Outputs written to %s", args.output_file)


if __name__ == "__main__":
    main()
