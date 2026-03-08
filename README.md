# Small Language Model

A comprehensive small language model implementation in TensorFlow, supporting text generation, question answering, sentiment analysis, and custom NLP tasks. Includes a custom transformer architecture from scratch and fine-tuning support for Hugging Face pre-trained models (GPT-2, BERT, DistilBERT, RoBERTa).

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training from Scratch](#training-from-scratch)
  - [Fine-tuning Pre-trained Models](#fine-tuning-pre-trained-models)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)

---

## Features

- **Custom Transformer**: GPT-style decoder-only transformer implemented from scratch in TensorFlow 2.x
- **Pre-trained Model Fine-tuning**: Fine-tune GPT-2, BERT, DistilBERT, RoBERTa from Hugging Face
- **Multiple Tasks**: Text generation, sentiment analysis, sequence classification, extractive Q&A
- **Efficient Training**: Gradient accumulation, early stopping, LR scheduling, mixed precision
- **Flexible Inference**: Top-K/Top-P sampling, beam search, greedy decoding, batch inference
- **GPU/TPU Support**: Automatic device detection and utilization

---

## Repository Structure

```
small-language-model/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml        # Model architecture settings
│   ├── training_config.yaml     # Training hyperparameters
│   └── inference_config.yaml    # Inference settings
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py        # Custom transformer implementation
│   │   ├── optimizations.py      # Memory & training optimization utilities
│   │   └── pretrained_wrapper.py # HuggingFace model wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset classes
│   │   └── preprocessing.py     # Text preprocessing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   └── metrics.py            # Metrics tracking
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py          # High-level inference interface
│       └── utils.py              # Sampling utilities
├── scripts/
│   ├── train.py                  # Train from scratch
│   ├── finetune_pretrained.py    # Fine-tune pre-trained models
│   ├── evaluate.py               # Model evaluation
│   ├── inference.py              # Run inference
│   └── download_model.py         # Download HuggingFace models
├── examples/
│   ├── text_generation_example.py
│   ├── qa_example.py
│   ├── sentiment_analysis_example.py
│   ├── large_model_example.py
│   └── finetuning_example.ipynb
└── tests/
    ├── __init__.py
    ├── test_model.py             # Model architecture tests
    ├── test_data.py              # Data pipeline tests
    └── test_training.py          # Training utilities tests
```

---

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install as a package (editable mode)

```bash
pip install -e .
```

---

## Quick Start

### Text Generation with GPT-2

```python
from src.model.pretrained_wrapper import PretrainedModelWrapper
from src.inference.predictor import Predictor

# Load pre-trained GPT-2
model = PretrainedModelWrapper(model_name="gpt2", task="text_generation")
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

predictor = Predictor(model=model, tokenizer=tokenizer, task="text_generation")

result = predictor.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)
print(result["generated_text"][0])
```

### Custom Transformer from Scratch

```python
from src.model.transformer import SmallTransformer, TransformerConfig
import tensorflow as tf

config = TransformerConfig(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    task="text_generation",
)
model = SmallTransformer(config)
print(f"Parameters: {model.count_parameters():,}")
```

---

## Usage

### Training from Scratch

```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --train_data data/train.txt \
    --val_data data/val.txt \
    --output_dir outputs/my_model
```

Or with Python:

```python
from src.model.transformer import SmallTransformer, TransformerConfig
from src.data.dataset import TextDataset
from src.training.trainer import Trainer
from transformers import AutoTokenizer
import tensorflow as tf

config = TransformerConfig(vocab_size=50257, d_model=512, num_heads=8, num_layers=6)
model = SmallTransformer(config)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = TextDataset("data/train.txt", tokenizer, max_seq_length=512)
train_ds = train_dataset.get_tf_dataset(batch_size=8, shuffle=True)

optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
trainer = Trainer(model=model, optimizer=optimizer, train_dataset=train_ds, num_epochs=10)
history = trainer.train()
```

### Fine-tuning Pre-trained Models

```bash
# Fine-tune GPT-2 on custom text
python scripts/finetune_pretrained.py \
    --model_name gpt2 \
    --task text_generation \
    --train_data data/train.txt \
    --output_dir outputs/finetuned_gpt2 \
    --num_epochs 3 \
    --learning_rate 2e-5

# Fine-tune BERT for sentiment analysis
python scripts/finetune_pretrained.py \
    --model_name bert-base-uncased \
    --task sentiment_analysis \
    --train_data data/sentiment_train.csv \
    --num_labels 2 \
    --output_dir outputs/bert_sentiment
```

### Inference

```bash
# Text generation
python scripts/inference.py \
    --model_path outputs/finetuned_gpt2 \
    --task text_generation \
    --prompt "The future of AI is"

# Sentiment analysis
python scripts/inference.py \
    --model_path outputs/bert_sentiment \
    --task sentiment_analysis \
    --input_file data/reviews.txt \
    --label_map '{"0": "negative", "1": "positive"}'

# Question answering
python scripts/inference.py \
    --model_path outputs/qa_model \
    --task question_answering \
    --question "What is the capital of France?" \
    --context "Paris is the capital and largest city of France."
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model_path outputs/finetuned_gpt2 \
    --test_data data/test.txt \
    --task text_generation \
    --output_file results/eval_results.json
```

### Download Pre-trained Models

```bash
# List available models
python scripts/download_model.py --list_models

# Download GPT-2
python scripts/download_model.py --model_name gpt2 --output_dir models/gpt2

# Download BERT for question answering
python scripts/download_model.py --model_name bert-base-uncased --task question_answering
```

---

## Model Architecture

The custom `SmallTransformer` is a GPT-style decoder-only transformer with:

- **Token Embeddings**: Learned or sinusoidal positional encodings
- **Transformer Blocks**: Multi-head self-attention + position-wise FFN with pre-norm
- **Attention**: Scaled dot-product multi-head attention with causal masking
- **Task Heads**: Language modeling head, classification head, Q&A span head

### Supported Tasks

| Task | Model Head | Loss |
|------|-----------|------|
| `text_generation` | LM Head (vocab projection) | Cross-entropy (next token) |
| `sentiment_analysis` | Classification Head | Cross-entropy |
| `sequence_classification` | Classification Head | Cross-entropy |
| `question_answering` | Span Head (start/end logits) | Cross-entropy (span) |

### Approximate Parameter Counts

| Config | d_model | Layers | Heads | Parameters |
|--------|---------|--------|-------|-----------|
| Small  | 256     | 4      | 8     | ~30M      |
| Medium | 512     | 6      | 8     | ~85M      |
| Large  | 768     | 12     | 12    | ~117M     |
| XL     | 1024    | 24     | 16    | ~345M     |
| 3B     | 2048    | 24     | 32    | ~3.0–3.5B |
| 5B     | 2560    | 32     | 32    | ~4.8–5.2B |

### Memory Requirements

| Config | Training (fp32) | Training (fp16 + grad ckpt) | Inference (fp16) |
|--------|----------------|----------------------------|-----------------|
| Large  | ~4 GB           | ~2 GB                      | ~1 GB           |
| XL     | ~14 GB          | ~7 GB                      | ~2.5 GB         |
| 3B     | ~192 GB         | ~96 GB                     | ~12 GB          |
| 5B     | ~320 GB         | ~160 GB                    | ~20 GB          |

> **Training 3B/5B models** requires enabling gradient checkpointing and
> mixed precision.  Use the predefined `CONFIG_3B` / `CONFIG_5B` constants
> which have these options enabled by default.

### Distributed Training Recommendations

| Model | Recommended Setup |
|-------|------------------|
| Large / XL | Single A100 40 GB |
| 3B | 2–4× A100 40 GB or 1× A100 80 GB, `MirroredStrategy` |
| 5B | 4× A100 40 GB or 2× A100 80 GB, `MirroredStrategy` or `MultiWorkerMirroredStrategy` |

```python
from src.model.transformer import SmallTransformer, CONFIG_3B
from src.model.optimizations import (
    enable_mixed_precision,
    apply_gradient_checkpointing,
    create_distribution_strategy,
    suggest_batch_size,
)

# 1. Enable fp16 before model creation
enable_mixed_precision()

# 2. Create a multi-GPU strategy
strategy = create_distribution_strategy("mirrored", num_gpus=4)

with strategy.scope():
    model = SmallTransformer(CONFIG_3B)

# 3. Enable gradient checkpointing
apply_gradient_checkpointing(model)

# 4. Auto-select batch size
batch_size = suggest_batch_size(3e9, seq_len=2048, mixed_precision=True)
print(f"Recommended batch size per GPU: {batch_size}")
```

---

## Configuration

### Model Config (`config/model_config.yaml`)

```yaml
model:
  type: transformer
  vocab_size: 50257
  d_model: 768
  num_heads: 12
  num_layers: 12
  d_ff: 3072
  max_seq_length: 1024
  dropout_rate: 0.1
  task: text_generation
```

### Training Config (`config/training_config.yaml`)

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  learning_rate: 5.0e-5
  weight_decay: 0.01
  early_stopping: true
  early_stopping_patience: 3
```

---

## Examples

Run the example scripts:

```bash
python examples/text_generation_example.py
python examples/sentiment_analysis_example.py
python examples/qa_example.py
python examples/large_model_example.py   # 3B/5B model demo
```

For a comprehensive Jupyter notebook walkthrough:

```bash
jupyter notebook examples/finetuning_example.ipynb
```

---

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_model.py -v     # Model architecture tests
pytest tests/test_data.py -v      # Data pipeline tests
pytest tests/test_training.py -v  # Training utility tests
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```
