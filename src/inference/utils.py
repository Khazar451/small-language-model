"""
Inference utility functions.

Low-level utilities for token sampling, beam search, and decoding strategies.
"""

import logging
from typing import Optional, List

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def top_k_top_p_filtering(
    logits: tf.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> tf.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) sampling.

    Args:
        logits: Logit tensor of shape (batch_size, vocab_size).
        top_k: If > 0, keep only the top-k highest probability tokens.
        top_p: If < 1.0, keep the smallest set of tokens whose cumulative
            probability exceeds top_p.
        filter_value: Value to assign to filtered tokens.
        min_tokens_to_keep: Minimum number of tokens to keep.

    Returns:
        Filtered logits tensor of the same shape.
    """
    logits = tf.cast(logits, tf.float32)

    if top_k > 0:
        top_k = max(top_k, min_tokens_to_keep)
        # Get top-k values and create a mask
        top_k_values, _ = tf.math.top_k(logits, k=min(top_k, logits.shape[-1]))
        threshold = top_k_values[:, -1:]  # (batch_size, 1)
        mask = logits < threshold
        logits = tf.where(mask, tf.fill(tf.shape(logits), filter_value), logits)

    if top_p < 1.0:
        sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
        sorted_probs = tf.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = tf.cumsum(sorted_probs, axis=-1, exclusive=True)

        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs > top_p

        # Ensure at least min_tokens_to_keep tokens
        sorted_mask = tf.concat(
            [
                tf.zeros_like(sorted_mask[:, :min_tokens_to_keep]),
                sorted_mask[:, min_tokens_to_keep:],
            ],
            axis=-1,
        )

        # Scatter sorted mask back to original ordering
        sorted_indices = tf.argsort(logits, direction="DESCENDING", axis=-1)
        indices_to_remove = tf.scatter_nd(
            indices=tf.stack([
                tf.repeat(tf.range(tf.shape(logits)[0]), tf.shape(logits)[1]),
                tf.reshape(sorted_indices, [-1]),
            ], axis=1),
            updates=tf.reshape(tf.cast(sorted_mask, tf.float32), [-1]),
            shape=tf.shape(logits),
        )
        logits = tf.where(
            tf.cast(indices_to_remove, tf.bool),
            tf.fill(tf.shape(logits), filter_value),
            logits,
        )

    return logits


def sample_generate(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None,
) -> tf.Tensor:
    """Generate tokens using the SmallTransformer model.

    Implements autoregressive generation with top-k/top-p sampling
    or greedy decoding.

    Args:
        model: SmallTransformer model instance.
        input_ids: Initial token IDs of shape (batch_size, prompt_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-K filtering parameter (0 = disabled).
        top_p: Nucleus sampling parameter (1.0 = disabled).
        do_sample: Whether to use sampling. If False, uses greedy decoding.
        pad_token_id: Padding token ID.
        eos_token_id: End-of-sequence token ID. Generation stops when generated.

    Returns:
        Generated token IDs including prompt, shape (batch_size, seq_len).
    """
    generated = input_ids

    for _ in range(max_new_tokens):
        outputs = model(generated, training=False)
        next_token_logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if do_sample:
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_tokens = tf.random.categorical(filtered_logits, num_samples=1, dtype=tf.int32)
        else:
            next_tokens = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            next_tokens = tf.expand_dims(next_tokens, axis=-1)

        generated = tf.concat([generated, next_tokens], axis=-1)

        if eos_token_id is not None:
            if tf.reduce_all(next_tokens == eos_token_id):
                break

    return generated


def greedy_decode(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    max_new_tokens: int = 100,
    eos_token_id: Optional[int] = None,
) -> tf.Tensor:
    """Greedy decoding: always select the most probable next token.

    Args:
        model: Language model with logits output.
        input_ids: Initial token IDs.
        max_new_tokens: Maximum tokens to generate.
        eos_token_id: Stop generation at this token.

    Returns:
        Generated token IDs.
    """
    return sample_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_token_id,
    )


def beam_search(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    beam_width: int = 5,
    max_new_tokens: int = 100,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> tf.Tensor:
    """Beam search decoding.

    Args:
        model: Language model with logits output.
        input_ids: Initial token IDs of shape (1, prompt_len).
        beam_width: Number of beams.
        max_new_tokens: Maximum tokens to generate.
        length_penalty: Penalty applied to sequence length.
        eos_token_id: Stop generation at this token.

    Returns:
        Best generated sequence as token IDs of shape (1, seq_len).
    """
    batch_size = tf.shape(input_ids)[0]
    if batch_size != 1:
        raise ValueError("Beam search currently only supports batch_size=1")

    # Initialize beams: list of (score, sequence) tuples
    beams = [(0.0, input_ids[0].numpy().tolist())]
    completed_beams = []

    for _ in range(max_new_tokens):
        all_candidates = []

        for score, seq in beams:
            seq_tensor = tf.constant([seq], dtype=tf.int32)
            outputs = model(seq_tensor, training=False)
            next_logits = outputs["logits"][0, -1, :]  # (vocab_size,)
            log_probs = tf.math.log_softmax(next_logits).numpy()

            # Get top beam_width candidates
            top_indices = np.argsort(log_probs)[-beam_width:]

            for idx in top_indices:
                new_score = score + log_probs[idx]
                new_seq = seq + [int(idx)]
                all_candidates.append((new_score, new_seq))

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = []

        for cand_score, cand_seq in all_candidates[:beam_width]:
            if eos_token_id is not None and cand_seq[-1] == eos_token_id:
                completed_beams.append((cand_score, cand_seq))
            else:
                beams.append((cand_score, cand_seq))

        if not beams:
            break

    # Return best sequence
    all_beams = beams + completed_beams
    if all_beams:
        best_seq = max(all_beams, key=lambda x: x[0] / (len(x[1]) ** length_penalty))[1]
    else:
        best_seq = beams[0][1] if beams else input_ids[0].numpy().tolist()

    return tf.constant([best_seq], dtype=tf.int32)
