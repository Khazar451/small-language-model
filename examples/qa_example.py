"""
Question Answering example using the small language model.

Demonstrates how to fine-tune and run inference for extractive Q&A.

Usage:
    python examples/qa_example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_qa_with_pretrained():
    """Demonstrate Q&A with a pre-trained model."""
    print("=== Question Answering with Pre-trained BERT ===\n")

    from src.model.pretrained_wrapper import PretrainedModelWrapper
    from src.inference.predictor import Predictor

    # Load BERT fine-tuned on SQuAD
    model = PretrainedModelWrapper(
        model_name="bert-base-uncased",
        task="question_answering",
    )
    tokenizer = model.tokenizer

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        task="question_answering",
        max_seq_length=384,
    )

    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "context": (
                "France is a country in Western Europe. "
                "Paris is the capital and largest city of France. "
                "It is known for its art, culture, and the Eiffel Tower."
            ),
        },
        {
            "question": "When was TensorFlow released?",
            "context": (
                "TensorFlow is an open-source machine learning framework developed by Google. "
                "It was released in November 2015. "
                "TensorFlow 2.0 was released in September 2019 with improved usability."
            ),
        },
    ]

    for qa in qa_pairs:
        result = predictor.answer_question(
            question=qa["question"],
            context=qa["context"],
        )
        print(f"Question: {result['question']}")
        print(f"Answer:   {result['answer']}")
        print(f"Score:    {result['score']:.4f}\n")


def example_qa_dataset():
    """Demonstrate creating and using a QA dataset."""
    print("=== QA Dataset Example ===\n")

    from transformers import AutoTokenizer
    from src.data.dataset import QADataset

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    contexts = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "Python was created by Guido van Rossum and released in 1991.",
        "The Amazon River is the largest river by discharge volume in the world.",
    ]
    questions = [
        "Where is the Eiffel Tower located?",
        "Who created Python?",
        "What is the Amazon River known for?",
    ]
    answers = [
        {"text": "Paris, France", "answer_start": 38},
        {"text": "Guido van Rossum", "answer_start": 18},
        {"text": "largest river by discharge volume", "answer_start": 16},
    ]

    dataset = QADataset(
        contexts=contexts,
        questions=questions,
        answers=answers,
        tokenizer=tokenizer,
        max_seq_length=384,
    )
    print(f"Created QA dataset with {len(dataset)} examples")

    tf_ds = dataset.get_tf_dataset(batch_size=2)
    for batch in tf_ds.take(1):
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Start positions: {batch['start_positions'].numpy()}")
        print(f"End positions:   {batch['end_positions'].numpy()}")


if __name__ == "__main__":
    print("Small Language Model - Question Answering Examples")
    print("=" * 50)
    example_qa_dataset()
