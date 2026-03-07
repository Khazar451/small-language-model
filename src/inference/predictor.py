"""
Inference and prediction utilities.

This module provides a Predictor class that wraps a trained model
for easy text generation, classification, and Q&A inference.
"""

import logging
import os
from typing import Optional, List, Dict, Any, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class Predictor:
    """High-level interface for model inference.

    Supports text generation, sentiment analysis, sequence classification,
    and extractive question answering.

    Args:
        model: Trained TensorFlow model.
        tokenizer: HuggingFace tokenizer.
        task: Task type ('text_generation', 'sequence_classification',
              'sentiment_analysis', 'question_answering').
        device: Device to run inference on ('cpu', 'gpu', or 'auto').
        max_seq_length: Maximum input sequence length.

    Example:
        >>> predictor = Predictor(model, tokenizer, task="text_generation")
        >>> result = predictor.predict("Once upon a time")
        >>> print(result["generated_text"])
    """

    def __init__(
        self,
        model: tf.keras.Model,
        tokenizer: Any,
        task: str = "text_generation",
        device: str = "auto",
        max_seq_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.max_seq_length = max_seq_length

        # Ensure pad token is set
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Predictor initialized for task '%s'", task)

    def predict(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference on one or more texts.

        Args:
            text: Input text or list of texts.
            **kwargs: Task-specific parameters forwarded to the appropriate method.

        Returns:
            Prediction result(s) as dictionary or list of dictionaries.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        if self.task == "text_generation":
            results = [self.generate(t, **kwargs) for t in texts]
        elif self.task in ("sequence_classification", "sentiment_analysis"):
            results = self.classify(texts, **kwargs)
        elif self.task == "question_answering":
            results = [self.answer_question(**t, **kwargs) if isinstance(t, dict) else t
                       for t in texts]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return results[0] if single else results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate text continuation from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.
            do_sample: Whether to use sampling.
            num_return_sequences: Number of sequences to generate.
            repetition_penalty: Penalty for repeating tokens.

        Returns:
            Dictionary with 'generated_text' and 'prompt'.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="tf",
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = inputs["input_ids"]

        # Use model.generate if available (HuggingFace models)
        if hasattr(self.model, "generate") or hasattr(
            getattr(self.model, "base_model", None), "generate"
        ):
            base = getattr(self.model, "base_model", self.model)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
            }
            generated_ids = base.generate(input_ids, **gen_kwargs)
        else:
            # Custom greedy/sampling generation for SmallTransformer
            from src.inference.utils import sample_generate
            generated_ids = sample_generate(
                model=self.model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or 0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            "prompt": prompt,
            "generated_text": generated_texts,
            "num_tokens_generated": int(generated_ids.shape[1]) - int(input_ids.shape[1]),
        }

    def classify(
        self,
        texts: List[str],
        label_map: Optional[Dict[int, str]] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Classify texts (sentiment analysis, sequence classification, etc.).

        Args:
            texts: List of input texts.
            label_map: Optional mapping from label indices to label names.
            batch_size: Batch size for inference.

        Returns:
            List of dicts with 'label', 'label_name', and 'scores'.
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
            )

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                training=False,
            )

            logits = outputs.get("logits")
            if logits is None:
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    raise ValueError("Model outputs do not contain 'logits'")

            probs = tf.nn.softmax(logits, axis=-1).numpy()
            predicted_labels = np.argmax(probs, axis=-1)

            for label, scores in zip(predicted_labels, probs):
                result = {
                    "label": int(label),
                    "scores": scores.tolist(),
                    "confidence": float(scores[label]),
                }
                if label_map:
                    result["label_name"] = label_map.get(int(label), str(label))
                all_results.append(result)

        return all_results

    def answer_question(
        self,
        question: str,
        context: str,
        max_answer_length: int = 100,
    ) -> Dict[str, Any]:
        """Answer an extractive question given a context.

        Args:
            question: The question to answer.
            context: The passage containing the answer.
            max_answer_length: Maximum number of tokens in the answer.

        Returns:
            Dictionary with 'answer', 'score', 'start', and 'end'.
        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="tf",
            truncation="only_second",
            padding="max_length",
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].numpy()

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            training=False,
        )

        start_logits = outputs.get("start_logits")
        end_logits = outputs.get("end_logits")

        if start_logits is None:
            raise ValueError(
                "Model does not support question answering (no start_logits/end_logits)"
            )

        start_logits = start_logits[0].numpy()
        end_logits = end_logits[0].numpy()

        # Find the best span
        best_score = float("-inf")
        best_start = 0
        best_end = 0

        for start in range(len(start_logits)):
            for end in range(start, min(start + max_answer_length, len(end_logits))):
                score = start_logits[start] + end_logits[end]
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_end = end

        # Convert token positions to character positions
        start_char = int(offset_mapping[best_start][0])
        end_char = int(offset_mapping[best_end][1])
        answer = context[start_char:end_char]

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "score": float(best_score),
            "start_char": start_char,
            "end_char": end_char,
        }

    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run batch inference on a list of texts.

        Args:
            texts: List of input texts.
            batch_size: Number of texts per batch.
            **kwargs: Additional parameters passed to predict.

        Returns:
            List of prediction results.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_results = self.predict(batch, **kwargs)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            logger.debug("Processed batch %d/%d", (i // batch_size) + 1,
                         (len(texts) + batch_size - 1) // batch_size)
        return results

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        task: str = "text_generation",
        num_labels: int = 2,
        **kwargs,
    ) -> "Predictor":
        """Load a predictor from a saved model directory.

        Args:
            model_path: Path to the saved model directory (custom SmallTransformer format).
            task: Task type.
            num_labels: Number of classification labels.
            **kwargs: Additional arguments passed to Predictor.

        Returns:
            Predictor instance.
        """
        from transformers import AutoTokenizer
        from src.model.transformer import SmallTransformer

        # Load the custom SmallTransformer model directly
        model = SmallTransformer.load_pretrained(model_path)

        # Try to load the tokenizer saved alongside the model;
        # fall back to the gpt2 tokenizer if none was saved there.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except OSError:
            logger.warning(
                "No tokenizer found at %s; falling back to 'gpt2' tokenizer.",
                model_path,
            )
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, task=task, **kwargs)
