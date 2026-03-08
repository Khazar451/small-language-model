"""Data package for loading and preprocessing datasets."""

from src.data.dataset import TextDataset, QADataset, ClassificationDataset, MultiFileTextDataset
from src.data.preprocessing import DataPreprocessor, tokenize_dataset
from src.data.streaming_dataset import StreamingTextDataset, collect_files
from src.data.huggingface_loader import HuggingFaceLoader, RECOMMENDED_DATASETS
from src.data.tokenizer_cache import TokenizerCache
from src.data.statistics import DataStatistics

__all__ = [
    "TextDataset",
    "QADataset",
    "ClassificationDataset",
    "MultiFileTextDataset",
    "DataPreprocessor",
    "tokenize_dataset",
    "StreamingTextDataset",
    "collect_files",
    "HuggingFaceLoader",
    "RECOMMENDED_DATASETS",
    "TokenizerCache",
    "DataStatistics",
]
