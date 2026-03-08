"""
Dataset classes for loading and serving training data.

This module provides TensorFlow Dataset wrappers for text generation,
question answering, and classification tasks.  It also provides
:class:`MultiFileTextDataset` for streaming from multiple files or
directories for large-scale pre-training.
"""

import logging
import os
from typing import Optional, List, Dict, Any, Tuple, Iterator, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class TextDataset:
    """Dataset for language model pre-training and text generation fine-tuning.

    Loads text data, tokenizes it, and creates overlapping chunks for
    next-token prediction training.

    Args:
        file_path: Path to the text file.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Maximum sequence length for each chunk.
        stride: Stride for sliding window over long documents.
            If None, defaults to max_seq_length (no overlap).
        pad_to_max_length: Whether to pad shorter sequences.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = TextDataset("data/train.txt", tokenizer, max_seq_length=512)
        >>> tf_dataset = dataset.get_tf_dataset(batch_size=8, shuffle=True)
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        max_seq_length: int = 512,
        stride: Optional[int] = None,
        pad_to_max_length: bool = False,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        self.pad_to_max_length = pad_to_max_length

        self.examples = self._load_and_tokenize()
        logger.info(
            "TextDataset loaded %d chunks from %s",
            len(self.examples), file_path,
        )

    def _load_and_tokenize(self) -> List[Dict[str, np.ndarray]]:
        """Load text file and create tokenized chunks.

        Returns:
            List of dictionaries with 'input_ids' and 'attention_mask'.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the full text
        encoding = self.tokenizer(
            text,
            return_tensors="np",
            truncation=False,
            add_special_tokens=True,
        )
        all_ids = encoding["input_ids"][0]

        examples = []
        total_len = len(all_ids)

        for start in range(0, total_len, self.stride):
            end = min(start + self.max_seq_length, total_len)
            chunk_ids = all_ids[start:end]

            if len(chunk_ids) < 2:
                continue

            if self.pad_to_max_length and len(chunk_ids) < self.max_seq_length:
                pad_len = self.max_seq_length - len(chunk_ids)
                pad_id = self.tokenizer.pad_token_id or 0
                mask = np.concatenate([
                    np.ones(len(chunk_ids), dtype=np.int32),
                    np.zeros(pad_len, dtype=np.int32),
                ])
                chunk_ids = np.concatenate([chunk_ids, np.full(pad_len, pad_id)])
            else:
                mask = np.ones(len(chunk_ids), dtype=np.int32)

            examples.append({
                "input_ids": chunk_ids.astype(np.int32),
                "attention_mask": mask,
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.examples[idx]

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        repeat: bool = False,
        buffer_size: int = 10000,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset from the examples.

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle the dataset.
            repeat: Whether to repeat the dataset indefinitely.
            buffer_size: Buffer size for shuffling.

        Returns:
            tf.data.Dataset yielding (input_ids, attention_mask) tuples.
        """
        if not self.examples:
            raise ValueError("No examples found in dataset")

        # Find the minimum sequence length for padding uniformity
        max_len = max(len(ex["input_ids"]) for ex in self.examples)

        def generator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            for ex in self.examples:
                yield ex["input_ids"], ex["attention_mask"]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Pad sequences in a batch to the same length
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.cast(self.tokenizer.pad_token_id or 0, tf.int32),
                tf.cast(0, tf.int32),
            ),
        )

        if repeat:
            dataset = dataset.repeat()

        return dataset.prefetch(tf.data.AUTOTUNE)


class ClassificationDataset:
    """Dataset for text classification tasks (sentiment analysis, etc.).

    Args:
        texts: List of input texts.
        labels: List of integer labels.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Maximum sequence length.
        label_map: Optional dictionary mapping string labels to integers.

    Example:
        >>> dataset = ClassificationDataset(
        ...     texts=["Great movie!", "Terrible film."],
        ...     labels=[1, 0],
        ...     tokenizer=tokenizer,
        ... )
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_seq_length: int = 128,
        label_map: Optional[Dict[str, int]] = None,
    ):
        assert len(texts) == len(labels), (
            f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})"
        )

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_map = label_map

        # Convert string labels if needed
        if label_map and isinstance(labels[0], str):
            labels = [label_map[lbl] for lbl in labels]
        self.labels = labels

        self.encodings = self._tokenize()
        logger.info("ClassificationDataset loaded %d examples", len(texts))

    def _tokenize(self) -> Dict[str, np.ndarray]:
        """Tokenize all texts.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' arrays.
        """
        encoding = self.tokenizer(
            self.texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return encoding

    def __len__(self) -> int:
        return len(self.texts)

    def get_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            repeat: Whether to repeat.

        Returns:
            tf.data.Dataset yielding ((input_ids, attention_mask), labels).
        """
        input_ids = tf.constant(self.encodings["input_ids"], dtype=tf.int32)
        attention_mask = tf.constant(self.encodings["attention_mask"], dtype=tf.int32)
        labels = tf.constant(self.labels, dtype=tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices((
            {"input_ids": input_ids, "attention_mask": attention_mask},
            labels,
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.texts))
        if repeat:
            dataset = dataset.repeat()

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class QADataset:
    """Dataset for extractive question answering tasks.

    Supports datasets in SQuAD format.

    Args:
        contexts: List of passage/context strings.
        questions: List of question strings.
        answers: List of answer dictionaries with 'text' and 'answer_start' keys.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Maximum sequence length.
        doc_stride: Stride for handling long contexts.

    Example:
        >>> dataset = QADataset(
        ...     contexts=["Paris is the capital of France."],
        ...     questions=["What is the capital of France?"],
        ...     answers=[{"text": "Paris", "answer_start": 0}],
        ...     tokenizer=tokenizer,
        ... )
    """

    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 384,
        doc_stride: int = 128,
    ):
        assert len(contexts) == len(questions) == len(answers)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

        self.examples = self._process_examples(contexts, questions, answers)
        logger.info("QADataset loaded %d examples", len(self.examples))

    def _process_examples(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Tokenize and find answer positions.

        Args:
            contexts: List of contexts.
            questions: List of questions.
            answers: List of answer dicts.

        Returns:
            List of processed example dicts.
        """
        examples = []
        for context, question, answer in zip(contexts, questions, answers):
            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_seq_length,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="np",
            )

            offset_mapping = encoding["offset_mapping"][0]
            answer_text = answer.get("text", "")
            answer_start_char = answer.get("answer_start", 0)
            answer_end_char = answer_start_char + len(answer_text)

            # Find token positions
            start_position = 0
            end_position = 0

            sequence_ids = encoding.sequence_ids(0)
            for i, (offset, seq_id) in enumerate(zip(offset_mapping, sequence_ids)):
                if seq_id != 1:
                    continue
                if offset[0] <= answer_start_char < offset[1]:
                    start_position = i
                if offset[0] < answer_end_char <= offset[1]:
                    end_position = i

            examples.append({
                "input_ids": encoding["input_ids"][0].astype(np.int32),
                "attention_mask": encoding["attention_mask"][0].astype(np.int32),
                "start_positions": np.array(start_position, dtype=np.int32),
                "end_positions": np.array(end_position, dtype=np.int32),
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def get_tf_dataset(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.

        Returns:
            tf.data.Dataset with inputs and answer positions.
        """
        input_ids = np.stack([ex["input_ids"] for ex in self.examples])
        attention_mask = np.stack([ex["attention_mask"] for ex in self.examples])
        start_positions = np.array([ex["start_positions"] for ex in self.examples])
        end_positions = np.array([ex["end_positions"] for ex in self.examples])

        dataset = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(input_ids),
            "attention_mask": tf.constant(attention_mask),
            "start_positions": tf.constant(start_positions),
            "end_positions": tf.constant(end_positions),
        })

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.examples))

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class MultiFileTextDataset:
    """Dataset for language model pre-training from multiple files or directories.

    A convenience wrapper around :class:`~src.data.streaming_dataset.StreamingTextDataset`
    that presents the same ``get_tf_dataset`` interface as :class:`TextDataset` while
    supporting multiple file formats and streaming to avoid loading all data
    into memory at once.

    Args:
        paths: File path(s), directory path(s), or glob pattern(s).
        tokenizer: HuggingFace-compatible tokenizer instance.
        max_seq_length: Maximum sequence length for each chunk.
        stride: Sliding-window stride (defaults to *max_seq_length*).
        recursive: Whether to recurse into subdirectories.
        shuffle: Whether to shuffle examples during iteration.
        shuffle_buffer_size: Buffer size for ``tf.data`` element shuffle.
        text_field: JSON key for text in ``.jsonl`` files.
        text_column: Column name for text in ``.parquet``/``.arrow`` files.
        extensions: File extensions to include (defaults to all supported).
        seed: Random seed for reproducible shuffling.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = MultiFileTextDataset(
        ...     paths=["data/books/", "data/web.jsonl"],
        ...     tokenizer=tokenizer,
        ...     max_seq_length=1024,
        ...     recursive=True,
        ... )
        >>> tf_ds = dataset.get_tf_dataset(batch_size=8)
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        tokenizer: Any,
        max_seq_length: int = 1024,
        stride: Optional[int] = None,
        recursive: bool = False,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10_000,
        text_field: str = "text",
        text_column: str = "text",
        extensions: Optional[List[str]] = None,
        seed: int = 42,
    ):
        from src.data.streaming_dataset import StreamingTextDataset  # noqa: PLC0415

        self._inner = StreamingTextDataset(
            paths=paths,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            stride=stride,
            recursive=recursive,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            text_field=text_field,
            text_column=text_column,
            extensions=extensions,
            seed=seed,
        )
        self.files = self._inner.files
        logger.info(
            "MultiFileTextDataset: %d source file(s)", len(self.files)
        )

    def get_tf_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        repeat: bool = False,
        buffer_size: int = 10_000,
    ) -> tf.data.Dataset:
        """Create a ``tf.data.Dataset`` from all source files.

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle examples.
            repeat: Whether to repeat the dataset indefinitely.
            buffer_size: Shuffle buffer size (ignored if *shuffle* is
                ``False``).

        Returns:
            ``tf.data.Dataset`` yielding ``(input_ids, attention_mask)``
            batches.
        """
        # Propagate shuffle setting to the inner dataset
        self._inner.shuffle = shuffle
        self._inner.shuffle_buffer_size = buffer_size
        return self._inner.get_tf_dataset(
            batch_size=batch_size,
            repeat=repeat,
            prefetch=True,
        )
