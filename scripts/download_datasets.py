#!/usr/bin/env python3
"""
Download popular large-scale datasets from the HuggingFace Hub.

Downloads one or more recommended pre-training datasets, saves them as
JSONL, and displays basic metadata.

Usage
-----
# List available datasets
python scripts/download_datasets.py --list

# Download OpenWebText (first 50 000 samples)
python scripts/download_datasets.py openwebtext --max-samples 50000 --output data/

# Download multiple datasets
python scripts/download_datasets.py openwebtext slimpajama --output data/
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def list_datasets() -> None:
    from src.data.huggingface_loader import RECOMMENDED_DATASETS

    print("\nAvailable pre-training datasets:\n")
    print(f"{'Key':<25} {'HF name':<40} {'Approx tokens'}")
    print("-" * 80)
    for key, info in RECOMMENDED_DATASETS.items():
        tokens = f"~{info['tokens_b']}B"
        print(f"{key:<25} {info['name']:<40} {tokens}")
    print()


def download_dataset(
    key: str,
    output_dir: str,
    max_samples: int,
    text_field: str,
    cache_dir: str,
) -> None:
    """Download a single dataset and save as JSONL."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        sys.exit(1)

    from src.data.huggingface_loader import RECOMMENDED_DATASETS

    if key in RECOMMENDED_DATASETS:
        info = RECOMMENDED_DATASETS[key]
        hf_name = info["name"]
        field = text_field or info["text_field"]
        logger.info("Downloading '%s' (%s) …", key, info["description"])
    else:
        hf_name = key
        field = text_field or "text"
        logger.info("Downloading '%s' …", hf_name)

    ds_kwargs: dict = {"streaming": True}
    if cache_dir:
        ds_kwargs["cache_dir"] = cache_dir

    dataset = load_dataset(hf_name, split="train", **ds_kwargs)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = key.replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_name}.jsonl")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get(field, "")
            if text and text.strip():
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    break
            if count % 10_000 == 0 and count > 0:
                logger.info("  Written %d documents …", count)

    logger.info("Saved %d documents to %s", count, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download HuggingFace pre-training datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset keys to download (use --list to see available keys).",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets.")
    parser.add_argument(
        "--output", default="data/", help="Output directory for JSONL files."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        dest="max_samples",
        help="Maximum number of documents per dataset (0 = unlimited).",
    )
    parser.add_argument(
        "--text-field",
        default="",
        dest="text_field",
        help="Override the text field name in the dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        dest="cache_dir",
        help="Cache directory for HuggingFace datasets.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.datasets:
        parser.print_help()
        sys.exit(1)

    for key in args.datasets:
        download_dataset(
            key=key,
            output_dir=args.output,
            max_samples=args.max_samples,
            text_field=args.text_field,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
