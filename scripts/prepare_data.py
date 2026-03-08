"""
Prepare and tokenize a raw text dataset for language model training.

Usage:
    python scripts/prepare_data.py \\
        --input_dir data/raw \\
        --output_dir data/processed \\
        --tokenizer gpt2 \\
        --max_seq_length 2048
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.data.preprocessing import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and tokenize a text dataset")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw text/jsonl files")
    parser.add_argument("--output_dir", required=True, help="Directory to write tokenized shards")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=100_000,
                        help="Number of token chunks per output shard")
    parser.add_argument("--val_fraction", type=float, default=0.01,
                        help="Fraction of data held out for validation (0 = no split)")
    parser.add_argument("--extension", default="*.txt",
                        help="Glob pattern for input files (default: *.txt)")
    return parser.parse_args()


def iter_texts(input_dir: str, extension: str):
    """Yield (file_path, text) pairs from *input_dir*."""
    import json

    pattern = os.path.join(input_dir, "**", extension)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        # Try jsonl as fallback
        paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True))
    logger.info("Found %d files matching pattern %s", len(paths), pattern)
    for path in paths:
        ext = os.path.splitext(path)[-1].lower()
        try:
            with open(path, encoding="utf-8") as fh:
                if ext == ".jsonl":
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                yield path, json.loads(line).get("text", "")
                            except json.JSONDecodeError:
                                continue
                else:
                    yield path, fh.read()
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    if args.val_fraction > 0:
        os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    seq_len = args.max_seq_length

    buffer = []
    shard_idx = 0
    total_chunks = 0
    val_buffer = []

    rng = np.random.default_rng(42)

    def flush_shard(data, split):
        nonlocal shard_idx
        path = os.path.join(args.output_dir, split, f"shard_{shard_idx:05d}.npz")
        arr = np.array(data, dtype=np.int32)
        np.savez_compressed(path, input_ids=arr)
        logger.info("Wrote %s with %d chunks", path, len(data))
        shard_idx += 1

    for _, text in iter_texts(args.input_dir, args.extension):
        text = clean_text(text)
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            buffer = buffer[seq_len:]
            is_val = args.val_fraction > 0 and rng.random() < args.val_fraction
            target = val_buffer if is_val else []
            if is_val:
                val_buffer.append(chunk[:seq_len])
            else:
                total_chunks += 1

            # Write train shards
            if not is_val:
                # Collect into a list to flush when shard is full
                if not hasattr(main, "_train_buf"):
                    main._train_buf = []
                main._train_buf.append(chunk[:seq_len])
                if len(main._train_buf) >= args.shard_size:
                    flush_shard(main._train_buf, "train")
                    main._train_buf = []

    # Flush remaining data
    if hasattr(main, "_train_buf") and main._train_buf:
        flush_shard(main._train_buf, "train")
    if val_buffer:
        flush_shard(val_buffer, "val")

    logger.info("Preprocessing complete. Train chunks written to %s", args.output_dir)


if __name__ == "__main__":
    main()
