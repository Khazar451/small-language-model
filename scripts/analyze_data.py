#!/usr/bin/env python3
"""
Generate statistics for training data (token distribution, vocabulary, etc.).

Reads from local files or a previously tokenized cache and writes a
human-readable JSON report.

Usage
-----
# Analyse local text files
python scripts/analyze_data.py --path data/train.txt --output data/statistics.json

# Analyse a directory recursively
python scripts/analyze_data.py --path data/ --recursive --output data/statistics.json

# Limit analysis to 10 000 texts
python scripts/analyze_data.py --path data/ --sample-size 10000
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute statistics for language model training data."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="File, directory, or glob pattern to analyse.",
    )
    parser.add_argument(
        "--output",
        default="data/statistics.json",
        help="Path to write the JSON statistics report.",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="HuggingFace tokenizer name (default: gpt2).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        dest="sample_size",
        help="Analyse only this many texts (0 = all).",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        dest="text_field",
        help="JSON field for text in .jsonl files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load tokenizer
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        logger.error("transformers required: pip install transformers")
        sys.exit(1)

    from src.data.streaming_dataset import StreamingTextDataset, collect_files
    from src.data.statistics import DataStatistics

    files = collect_files(args.path, recursive=args.recursive)
    if not files:
        logger.error("No supported files found in: %s", args.path)
        sys.exit(1)

    logger.info("Analysing %d file(s) …", len(files))

    def _stream_texts():
        ds = StreamingTextDataset(
            paths=args.path,
            tokenizer=tokenizer,  # tokenizer only used for chunking; we stream raw text here
            max_seq_length=512,
            recursive=args.recursive,
            shuffle=False,
            text_field=args.text_field,
        )
        yield from ds.stream_texts()

    stats = DataStatistics(tokenizer, output_path=args.output)
    result = stats.analyze_texts(
        _stream_texts(),
        sample_size=args.sample_size or None,
    )

    if not result:
        logger.error("No data was analysed.")
        sys.exit(1)

    stats.save()

    print("\n=== Data Statistics ===")
    print(f"  Texts analysed : {result['num_texts']:,}")
    print(f"  Empty texts    : {result['num_empty']:,}")
    print(f"  Total tokens   : {result['total_tokens']:,}")
    tpt = result["tokens_per_text"]
    print(f"  Tokens / text  : mean={tpt['mean']:.1f}  median={tpt['median']:.1f}  "
          f"min={tpt['min']}  max={tpt['max']}")
    print(f"  Unique tokens  : {result['vocabulary']['unique_tokens_seen']:,}")
    print(f"\nReport saved to: {args.output}\n")


if __name__ == "__main__":
    main()
