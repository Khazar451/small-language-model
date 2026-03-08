"""
Download public datasets from HuggingFace Hub for language model training.

Usage:
    python scripts/download_datasets.py --dataset openwebtext --output_dir data/
    python scripts/download_datasets.py --dataset wikitext --config wikitext-103-raw-v1 --output_dir data/
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Curated list of well-known datasets suitable for language model pre-training
AVAILABLE_DATASETS = {
    "wikitext": {
        "config": "wikitext-103-raw-v1",
        "text_field": "text",
        "description": "English Wikipedia text (103M tokens)",
    },
    "openwebtext": {
        "config": None,
        "text_field": "text",
        "description": "Open-source reproduction of WebText (~38GB)",
    },
    "c4": {
        "config": "en",
        "text_field": "text",
        "description": "Colossal Clean Crawled Corpus (~750GB)",
    },
    "pile": {
        "config": None,
        "text_field": "text",
        "description": "The Pile — 825GB diverse text corpus",
    },
    "cc_news": {
        "config": None,
        "text_field": "text",
        "description": "CommonCrawl news articles",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets for LM training")
    parser.add_argument("--dataset", help="Dataset name (see --list for available options)")
    parser.add_argument("--config", default=None, help="Dataset config/subset name (overrides default)")
    parser.add_argument("--split", default="train", help="Dataset split to download (default: train)")
    parser.add_argument("--output_dir", default="data", help="Directory to save downloaded files")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream rather than download fully (saves disk space)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to download")
    return parser.parse_args()


def list_datasets() -> None:
    print("\nAvailable datasets:")
    print(f"  {'Name':<20} {'Description'}")
    print("  " + "-" * 60)
    for name, meta in AVAILABLE_DATASETS.items():
        print(f"  {name:<20} {meta['description']}")
    print()


def download(
    dataset_name: str,
    config: str,
    split: str,
    output_dir: str,
    streaming: bool,
    max_examples: int,
    text_field: str,
) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.error("Please install the 'datasets' package: pip install datasets")
        sys.exit(1)

    logger.info("Loading dataset '%s' (config=%s, split=%s)", dataset_name, config, split)
    ds = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=streaming,
        trust_remote_code=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_name}_{split}.jsonl")

    import json

    count = 0
    with open(out_file, "w", encoding="utf-8") as fh:
        for example in ds:
            text = example.get(text_field, "")
            if not text:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if max_examples and count >= max_examples:
                break
            if count % 10_000 == 0:
                logger.info("  Written %d examples…", count)

    logger.info("Download complete: %d examples saved to %s", count, out_file)


def main() -> None:
    args = parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        logger.error("Provide --dataset or --list")
        sys.exit(1)

    meta = AVAILABLE_DATASETS.get(args.dataset, {"config": None, "text_field": "text"})
    config = args.config or meta.get("config")
    text_field = meta.get("text_field", "text")

    download(
        dataset_name=args.dataset,
        config=config,
        split=args.split,
        output_dir=args.output_dir,
        streaming=args.streaming,
        max_examples=args.max_examples,
        text_field=text_field,
    )


if __name__ == "__main__":
    main()
