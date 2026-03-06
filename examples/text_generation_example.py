"""
Text generation example using the small language model.

Demonstrates how to generate text with a trained or pre-trained model.

Usage:
    python examples/text_generation_example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_with_custom_model():
    """Generate text with the custom SmallTransformer."""
    import tensorflow as tf
    from src.model.transformer import SmallTransformer, TransformerConfig

    print("=== Text Generation with Custom Transformer ===\n")

    # Create a small model for demonstration
    config = TransformerConfig(
        vocab_size=50257,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_length=512,
        task="text_generation",
    )
    model = SmallTransformer(config)

    # Load GPT-2 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate text using the predictor
    from src.inference.predictor import Predictor
    predictor = Predictor(model=model, tokenizer=tokenizer, task="text_generation")

    prompts = [
        "Once upon a time in a land far away,",
        "The future of artificial intelligence",
        "In the beginning of the universe,",
    ]

    for prompt in prompts:
        result = predictor.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {result['generated_text'][0]}\n")


def example_with_pretrained_gpt2():
    """Generate text with fine-tuned GPT-2."""
    print("=== Text Generation with Pre-trained GPT-2 ===\n")

    from src.model.pretrained_wrapper import PretrainedModelWrapper
    from src.inference.predictor import Predictor

    model = PretrainedModelWrapper(model_name="gpt2", task="text_generation")
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    predictor = Predictor(model=model, tokenizer=tokenizer, task="text_generation")

    prompts = [
        "The weather today is",
        "Machine learning is",
    ]

    for prompt in prompts:
        result = predictor.generate(
            prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_k=40,
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {result['generated_text'][0]}\n")


if __name__ == "__main__":
    print("Small Language Model - Text Generation Examples")
    print("=" * 50)
    example_with_custom_model()
