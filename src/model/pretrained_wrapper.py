"""
Wrapper for pre-trained Hugging Face models.

This module provides a TensorFlow-compatible wrapper around Hugging Face
pre-trained models (BERT, GPT-2, DistilBERT, etc.) for fine-tuning on
custom tasks.
"""

import logging
import os
from typing import Optional, Dict, Any, List

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Map of supported model names to their HuggingFace identifiers
SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilgpt2": "distilgpt2",
    "roberta-base": "roberta-base",
}


class PretrainedModelWrapper(tf.keras.Model):
    """Wrapper for Hugging Face pre-trained models.

    This class provides a unified interface for fine-tuning pre-trained
    language models (BERT, GPT-2, DistilBERT, RoBERTa) on various NLP tasks.

    Args:
        model_name: Name or path of the pre-trained model.
        task: Task type ('text_generation', 'sequence_classification',
              'question_answering', 'sentiment_analysis').
        num_labels: Number of output labels for classification tasks.
        freeze_base: Whether to freeze the base model weights during fine-tuning.
        num_frozen_layers: Number of layers to freeze from the bottom.

    Example:
        >>> wrapper = PretrainedModelWrapper("gpt2", task="text_generation")
        >>> input_ids = tf.constant([[1, 2, 3, 4, 5]])
        >>> outputs = wrapper(input_ids)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        task: str = "text_generation",
        num_labels: int = 2,
        freeze_base: bool = False,
        num_frozen_layers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels
        self.freeze_base = freeze_base
        self.num_frozen_layers = num_frozen_layers

        self._load_model(model_name, task, num_labels)

        if freeze_base:
            self._freeze_base_model()
        elif num_frozen_layers > 0:
            self._freeze_layers(num_frozen_layers)

        logger.info(
            "Loaded pre-trained model '%s' for task '%s'",
            model_name, task,
        )

    def _load_model(self, model_name: str, task: str, num_labels: int) -> None:
        """Load the pre-trained model from HuggingFace.

        Args:
            model_name: HuggingFace model identifier.
            task: Task type to determine the model head.
            num_labels: Number of output labels.
        """
        try:
            from transformers import (
                TFAutoModelForCausalLM,
                TFAutoModelForSequenceClassification,
                TFAutoModelForQuestionAnswering,
                AutoTokenizer,
                TFAutoModel,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required to use pre-trained models. "
                "Install it with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if task == "text_generation":
            self.base_model = TFAutoModelForCausalLM.from_pretrained(model_name)
        elif task in ("sequence_classification", "sentiment_analysis"):
            self.base_model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        elif task == "question_answering":
            self.base_model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
        else:
            # Load base model for custom heads
            self.base_model = TFAutoModel.from_pretrained(model_name)
            self._build_custom_head(task, num_labels)

    def _build_custom_head(self, task: str, num_labels: int) -> None:
        """Build a custom task head on top of the base model.

        Args:
            task: Task type.
            num_labels: Number of output labels.
        """
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_labels, name="custom_classifier")
        logger.info("Built custom task head for task '%s'", task)

    def _freeze_base_model(self) -> None:
        """Freeze all base model weights."""
        self.base_model.trainable = False
        logger.info("Frozen all base model weights")

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N layers of the base model.

        Args:
            num_layers: Number of layers to freeze from the bottom.
        """
        frozen = 0
        for layer in self.base_model.layers:
            if frozen >= num_layers:
                break
            layer.trainable = False
            frozen += 1
        logger.info("Frozen %d layers of the base model", frozen)

    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Forward pass through the pre-trained model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            token_type_ids: Token type IDs for BERT-style models.
            training: Whether in training mode.

        Returns:
            Dictionary of model outputs (task-specific).
        """
        kwargs = {
            "input_ids": input_ids,
            "training": training,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.base_model(**kwargs)

        if hasattr(self, "classifier"):
            # Custom head: use CLS token or last hidden state
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state[:, 0, :]
            else:
                hidden = outputs[0][:, 0, :]
            hidden = self.dropout(hidden, training=training)
            logits = self.classifier(hidden)
            return {"logits": logits, "hidden_states": hidden}

        return {k: v for k, v in outputs.items() if v is not None}

    def compute_loss(
        self,
        input_ids: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        start_positions: Optional[tf.Tensor] = None,
        end_positions: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Compute task-specific loss.

        Args:
            input_ids: Input token IDs.
            labels: Labels for supervised tasks.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs for BERT.
            start_positions: Start positions for Q&A.
            end_positions: End positions for Q&A.
            training: Whether in training mode.

        Returns:
            Dictionary with 'loss' and task-specific outputs.
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "training": training,
        }

        if self.task == "text_generation" and labels is not None:
            kwargs["labels"] = labels
        elif self.task in ("sequence_classification", "sentiment_analysis") and labels is not None:
            kwargs["labels"] = labels
        elif self.task == "question_answering":
            if start_positions is not None:
                kwargs["start_positions"] = start_positions
            if end_positions is not None:
                kwargs["end_positions"] = end_positions

        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        outputs = self.base_model(**kwargs)

        result = {}
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
        if hasattr(outputs, "logits") and outputs.logits is not None:
            result["logits"] = outputs.logits
        if hasattr(outputs, "start_logits"):
            result["start_logits"] = outputs.start_logits
        if hasattr(outputs, "end_logits"):
            result["end_logits"] = outputs.end_logits

        # Handle custom head
        if hasattr(self, "classifier") and labels is not None:
            forward_out = self.call(input_ids, attention_mask, token_type_ids, training)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            result["loss"] = loss_fn(labels, forward_out["logits"])
            result["logits"] = forward_out["logits"]

        return result

    def generate(
        self,
        input_ids: tf.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> tf.Tensor:
        """Generate text using the pre-trained model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Number of top tokens to consider.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling.
            num_return_sequences: Number of sequences to return.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID.

        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, seq_len).
        """
        if not hasattr(self.base_model, "generate"):
            raise NotImplementedError(
                f"The model '{self.model_name}' does not support text generation."
            )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
        }

        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id

        return self.base_model.generate(input_ids, **gen_kwargs)

    def save_pretrained(self, save_dir: str) -> None:
        """Save the fine-tuned model and tokenizer.

        Args:
            save_dir: Directory to save to.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Pre-trained model saved to %s", save_dir)

    @classmethod
    def load_finetuned(
        cls,
        load_dir: str,
        task: str = "text_generation",
        num_labels: int = 2,
    ) -> "PretrainedModelWrapper":
        """Load a fine-tuned model from disk.

        Args:
            load_dir: Directory containing the saved model.
            task: Task type.
            num_labels: Number of labels for classification.

        Returns:
            Loaded PretrainedModelWrapper instance.
        """
        return cls(
            model_name=load_dir,
            task=task,
            num_labels=num_labels,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters.

        Returns:
            Dictionary with 'trainable' and 'total' parameter counts.
        """
        trainable = sum(np.prod(v.shape) for v in self.trainable_variables)
        total = sum(np.prod(v.shape) for v in self.variables)
        return {"trainable": trainable, "total": total}
