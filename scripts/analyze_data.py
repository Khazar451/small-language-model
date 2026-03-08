"""
Analyze a text dataset and print statistics.

Usage:
    python scripts/analyze_data.py --input_dir data/processed --tokenizer gpt2
    python scripts/analyze_data.py --file data/train.txt --tokenizer gpt2
"""

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.data.statistics import DatasetStatistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and report dataset statistics")
    parser.add_argument("--input_dir", default=None, help="Directory with .txt or .jsonl files")
    parser.add_argument("--file", default=None, help="Single file to analyze")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--sample_size", type=int, default=10_000,
                        help="Max documents to analyze (0 = all)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Sequence length for chunk estimation")
    return parser.parse_args()


def load_texts(input_dir: str = None, file: str = None) -> list:
    """Return a list of text strings from files."""
    import json

    texts = []
    paths = []

    if file:
        paths = [file]
    elif input_dir:
        for ext in ("*.txt", "*.jsonl"):
            paths.extend(sorted(glob.glob(os.path.join(input_dir, "**", ext), recursive=True)))

    for path in paths:
        ext = os.path.splitext(path)[-1].lower()
        try:
            with open(path, encoding="utf-8") as fh:
                if ext == ".jsonl":
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                texts.append(json.loads(line).get("text", ""))
                            except json.JSONDecodeError:
                                continue
                else:
                    texts.append(fh.read())
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)

    logger.info("Loaded %d documents from %d files", len(texts), len(paths))
    return texts


def main() -> None:
    args = parse_args()

    if not args.input_dir and not args.file:
        logger.error("Provide --input_dir or --file")
        sys.exit(1)

    texts = load_texts(args.input_dir, args.file)
    if not texts:
        logger.error("No text documents found")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sample_size = args.sample_size if args.sample_size > 0 else None

    stats = DatasetStatistics(tokenizer=tokenizer, sample_size=sample_size)
    report = stats.compute(texts)
    DatasetStatistics.print_report(report)

    est = stats.estimate_training_tokens(texts, max_seq_length=args.max_seq_length)
    print(f"\nTraining chunk estimate (seq_len={args.max_seq_length}):")
    print(f"  Total tokens  : {est['total_tokens']:,}")
    print(f"  Num chunks    : {est['num_chunks']:,}")
    print(f"  Steps / epoch : {est['estimated_steps_per_epoch']:,} (batch_size=1)")


if __name__ == "__main__":
    main()
