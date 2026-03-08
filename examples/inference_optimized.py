"""
Optimized inference example using production-grade SmallTransformer features.

Demonstrates:
- Loading a model with GQA + RoPE + SwiGLU
- INT8 / INT4 weight quantization
- KV-cache aware generation (top-K sampling)
- Latency and throughput benchmarking

Usage:
    python examples/inference_optimized.py

Requirements:
    pip install tensorflow transformers
"""

import os
import sys
import time
import logging

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.transformer import SmallTransformer, TransformerConfig
from src.model.quantization import (
    quantize_model_weights,
    estimate_quantized_size_gb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_optimized_model(task: str = "text_generation") -> SmallTransformer:
    """Build a small optimized model suitable for a consumer GPU."""
    config = TransformerConfig(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_kv_heads=2,       # GQA: 2 KV heads (6x KV cache reduction vs MHA)
        num_layers=12,
        d_ff=3072,
        max_seq_length=2048,
        task=task,
        use_gqa=True,
        use_rope=True,
        use_swiglu=True,
        use_flash_attention=False,
    )
    model = SmallTransformer(config)
    # Build by running a dummy forward pass
    _ = model(tf.zeros((1, 1), dtype=tf.int32))
    return model


# ---------------------------------------------------------------------------
# Text generation (greedy / top-K)
# ---------------------------------------------------------------------------

def generate(
    model: SmallTransformer,
    input_ids: tf.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> tf.Tensor:
    """Autoregressive generation with optional top-K sampling.

    Args:
        model: A built SmallTransformer model.
        input_ids: Prompt token IDs of shape (batch, seq_len).
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature (1.0 = unchanged, <1 = sharper).
        top_k: Keep only top-K logits before sampling (0 = greedy).

    Returns:
        Token IDs tensor of shape (batch, seq_len + max_new_tokens).
    """
    generated = input_ids
    for _ in range(max_new_tokens):
        outputs = model(generated, training=False)
        logits = outputs["logits"][:, -1, :]  # last token logits

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            top_values, _ = tf.math.top_k(logits, k=top_k)
            min_val = top_values[:, -1:]  # minimum of top-k values
            logits = tf.where(logits < min_val, tf.fill(tf.shape(logits), -1e9), logits)

        next_token = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        generated = tf.concat([generated, next_token], axis=1)

    return generated


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: SmallTransformer,
    batch_size: int = 1,
    seq_len: int = 128,
    num_runs: int = 20,
    warmup: int = 5,
) -> dict:
    """Measure inference latency and throughput.

    Args:
        model: A built SmallTransformer.
        batch_size: Number of sequences per batch.
        seq_len: Input sequence length.
        num_runs: Number of timed runs after warm-up.
        warmup: Number of warm-up runs (not timed).

    Returns:
        Dict with 'mean_ms', 'std_ms', 'tokens_per_second'.
    """
    dummy = tf.zeros((batch_size, seq_len), dtype=tf.int32)
    for _ in range(warmup):
        _ = model(dummy, training=False)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    tokens_per_sec = (batch_size * seq_len) / (mean_ms / 1000)

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "tokens_per_second": tokens_per_sec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Build optimized model
    logger.info("Building optimized model ...")
    model = build_optimized_model()
    param_count = model.count_parameters()
    logger.info("Parameters: %d (%.2fB)", param_count, param_count / 1e9)

    fp32_size_gb = param_count * 4 / (1024 ** 3)
    fp16_size_gb = param_count * 2 / (1024 ** 3)
    logger.info("FP32 size: %.2f GB | FP16 size: %.2f GB", fp32_size_gb, fp16_size_gb)

    # 2. Run quantization analysis
    logger.info("\n--- Weight Quantization ---")
    for mode in ("int8", "int4"):
        quant_results = quantize_model_weights(model, mode=mode)
        est_gb = estimate_quantized_size_gb(model, mode=mode)
        logger.info(
            "Mode=%s | Layers quantized=%d | Estimated size=%.3f GB",
            mode, len(quant_results), est_gb,
        )

    # 3. Latency benchmark
    logger.info("\n--- Inference Latency Benchmark ---")
    for seq_len in (64, 128, 256):
        metrics = benchmark_inference(model, batch_size=1, seq_len=seq_len, num_runs=10)
        logger.info(
            "seq_len=%d | mean=%.1f ms ± %.1f ms | throughput=%.0f tok/s",
            seq_len, metrics["mean_ms"], metrics["std_ms"], metrics["tokens_per_second"],
        )

    # 4. Example generation
    logger.info("\n--- Text Generation Demo ---")
    prompt = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)  # toy token IDs
    generated = generate(model, prompt, max_new_tokens=20, top_k=50)
    logger.info("Generated token IDs (shape=%s): %s", generated.shape, generated.numpy().tolist())


if __name__ == "__main__":
    main()
