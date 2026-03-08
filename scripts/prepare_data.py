#!/usr/bin/env python3
"""
Download and prepare datasets for large-scale language model training.

Supports local file preparation and downloading from the HuggingFace Hub.

Usage
-----
# Prepare a local text file / directory
python scripts/prepare_data.py --source local --path data/raw/ --output data/prepared/

# Download an HuggingFace dataset and save a sample as JSONL
python scripts/prepare_data.py \\
    --source huggingface \\
    --dataset openwebtext \\
    --split train \\
    --max-samples 100000 \\
    --output data/openwebtext_100k.jsonl
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_local(args: argparse.Namespace) -> None:
    """Clean and split a local text file or directory."""
    from src.data.preprocessing import DataPreprocessor
    from src.data.streaming_dataset import collect_files

    files = collect_files(args.path, recursive=args.recursive)
    if not files:
        logger.error("No supported files found in: %s", args.path)
        sys.exit(1)

    preprocessor = DataPreprocessor(
        lowercase=args.lowercase,
        remove_special_chars=False,
    )

    texts = []
    for fpath in files:
        logger.info("Reading %s", fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(preprocessor.clean_text(line))

    if args.deduplicate:
        texts = preprocessor.deduplicate(texts)

    if args.min_length or args.max_length:
        texts = preprocessor.filter_by_length(
            texts,
            min_length=args.min_length or 0,
            max_length=args.max_length or 10**9,
        )

    os.makedirs(args.output, exist_ok=True)
    train, val, test = preprocessor.split_dataset(texts, ratios=(0.8, 0.1, 0.1))
    preprocessor.save_splits(train, val, test, args.output)
    logger.info("Wrote splits to %s", args.output)


def prepare_huggingface(args: argparse.Namespace) -> None:
    """Download an HuggingFace dataset and save it as JSONL."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        sys.exit(1)

    from src.data.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    logger.info("Loading dataset '%s' split '%s'", args.dataset, args.split)

    subset = args.subset or None
    ds_kwargs = {"streaming": True}
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir

    if subset:
        dataset = load_dataset(args.dataset, subset, split=args.split, **ds_kwargs)
    else:
        dataset = load_dataset(args.dataset, split=args.split, **ds_kwargs)

    text_field = args.text_field or "text"
    output_path = args.output
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"{args.dataset.replace('/', '_')}.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for example in dataset:
            text = example.get(text_field, "")
            if not text or not text.strip():
                continue
            text = preprocessor.clean_text(text)
            if not text:
                continue
            out_f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
            if count % 10_000 == 0:
                logger.info("Written %d documents …", count)

    logger.info("Saved %d documents to %s", count, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for LM training."
    )
    sub = parser.add_subparsers(dest="source", required=True)

    # Local sub-command
    local = sub.add_parser("local", help="Prepare a local file or directory.")
    local.add_argument("--path", required=True, help="File or directory path.")
    local.add_argument("--output", required=True, help="Output directory for splits.")
    local.add_argument("--recursive", action="store_true")
    local.add_argument("--lowercase", action="store_true")
    local.add_argument("--deduplicate", action="store_true")
    local.add_argument("--min-length", type=int, default=50, dest="min_length")
    local.add_argument("--max-length", type=int, default=0, dest="max_length")

    # HuggingFace sub-command
    hf = sub.add_parser("huggingface", help="Download a HuggingFace dataset.")
    hf.add_argument("--dataset", required=True, help="Dataset name (e.g. openwebtext).")
    hf.add_argument("--split", default="train")
    hf.add_argument("--subset", default=None)
    hf.add_argument("--text-field", default="text", dest="text_field")
    hf.add_argument("--output", required=True, help="Output JSONL file or directory.")
    hf.add_argument("--max-samples", type=int, default=0, dest="max_samples")
    hf.add_argument("--cache-dir", default=None, dest="cache_dir")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "local":
        prepare_local(args)
    else:
        prepare_huggingface(args)


if __name__ == "__main__":
    main()
