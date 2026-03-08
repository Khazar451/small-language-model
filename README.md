# Small Language Model

A comprehensive small language model implementation in TensorFlow, supporting text generation, question answering, sentiment analysis, and custom NLP tasks. Includes a custom transformer architecture from scratch (scaling from ~30M to 8B parameters) and fine-tuning support for Hugging Face pre-trained models (GPT-2, BERT, DistilBERT, RoBERTa).

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Sizes & Memory Requirements](#model-sizes--memory-requirements)
- [Architecture Optimizations](#architecture-optimizations)
- [Usage](#usage)
  - [Training from Scratch](#training-from-scratch)
  - [Fine-tuning Pre-trained Models](#fine-tuning-pre-trained-models)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Production-Grade Features](#production-grade-features)
  - [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
  - [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
  - [SwiGLU Feed-Forward Network](#swiglu-feed-forward-network)
  - [Flash Attention](#flash-attention)
  - [Gradient Checkpointing](#gradient-checkpointing)
  - [Mixed Precision Training](#mixed-precision-training)
  - [Quantization](#quantization)
  - [Distributed Training](#distributed-training)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Examples](#examples)
- [Testing](#testing)

---

## Features

- **Custom Transformer**: GPT-style decoder-only transformer implemented from scratch in TensorFlow 2.x, scaling from 30M to 8B parameters
- **Production-Grade Optimizations**: Grouped-Query Attention, RoPE, SwiGLU, Flash Attention, gradient checkpointing, mixed precision
- **Pre-trained Model Fine-tuning**: Fine-tune GPT-2, BERT, DistilBERT, RoBERTa from Hugging Face
- **Multiple Tasks**: Text generation, sentiment analysis, sequence classification, extractive Q&A
- **Efficient Training**: Gradient accumulation, early stopping, LR scheduling, mixed precision (FP16/BF16)
- **Flexible Inference**: Top-K/Top-P sampling, beam search, greedy decoding, batch inference
- **Quantization**: INT8 and INT4 weight quantization for local deployment
- **Distributed Training**: Multi-GPU (MirroredStrategy), TPU support, DeepSpeed config
- **GPU/TPU Support**: Automatic device detection and utilization
- **Large-Scale Data Pipeline**: Streaming datasets, multi-source loading, HuggingFace Hub integration, tokenization caching, and quality filtering

---

## Repository Structure

```
small-language-model/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml        # Model architecture settings (inc. 1B-8B presets)
│   ├── training_config.yaml     # Training hyperparameters
│   ├── inference_config.yaml    # Inference settings
│   └── data_config.yaml         # Data pipeline configuration
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py        # Custom transformer (inc. GQA/RoPE/SwiGLU)
│   │   ├── optimizations.py      # GQA, RoPE, SwiGLU, Flash Attention modules
│   │   ├── quantization.py       # INT8/INT4 quantization utilities
│   │   └── pretrained_wrapper.py # HuggingFace model wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset classes (incl. MultiFileTextDataset)
│   │   ├── preprocessing.py      # Text preprocessing & quality filtering
│   │   ├── streaming_dataset.py  # Streaming multi-file/dir data loader
│   │   ├── huggingface_loader.py # HuggingFace Hub integration
│   │   ├── tokenizer_cache.py    # Tokenization caching to disk
│   │   └── statistics.py         # Data analysis and monitoring
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop (with throughput tracking)
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
│   ├── download_model.py         # Download HuggingFace models
│   ├── prepare_data.py           # Download & prepare datasets
│   ├── download_datasets.py      # Download popular LM datasets
│   └── analyze_data.py           # Generate data statistics
├── examples/
│   ├── text_generation_example.py
│   ├── qa_example.py
│   ├── sentiment_analysis_example.py
│   ├── finetuning_example.ipynb
│   ├── train_3b_model.py          # 3B model training example
│   └── inference_optimized.py     # Optimized inference with quantization
└── tests/
    ├── __init__.py
    ├── test_model.py             # Model architecture tests
    ├── test_data.py              # Data pipeline tests
    ├── test_training.py          # Training utilities tests
    └── test_optimizations.py     # GQA, RoPE, SwiGLU, quantization tests
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

### Production-Grade 3B Model (Quick Start)

```python
from src.model.transformer import SmallTransformer

# Create a 3B model with GQA + RoPE + SwiGLU (all optimizations enabled)
model = SmallTransformer.for_size("3b")
print(f"Parameters: {model.count_parameters():,}")   # ~3B
print(f"Config: {model.config.to_dict()}")
```

---

## Model Sizes & Memory Requirements

| Model | Parameters | d_model | Heads | KV Heads | Layers | d_ff | FP16 Size | INT4 Size |
|-------|-----------|---------|-------|----------|--------|------|-----------|-----------|
| **1B** | ~1.3B | 768 | 12 | 2 | 12 | 3072 | ~2.5 GB | ~700 MB |
| **3B** | ~3.0B | 1536 | 16 | 4 | 24 | 6144 | ~6 GB | ~1.5 GB |
| **5B** | ~5.2B | 2048 | 32 | 8 | 32 | 8192 | ~10 GB | ~2.5 GB |
| **8B** | ~8.0B | 2560 | 32 | 8 | 40 | 10240 | ~16 GB | ~4 GB |

*All predefined configs use GQA + RoPE + SwiGLU for maximum efficiency.*

---

## Architecture Optimizations

### Grouped-Query Attention (GQA)

Replaces standard multi-head attention (MHA) to reduce KV cache memory during inference. Each group of query heads shares a single key/value head.

```python
from src.model.transformer import TransformerConfig, SmallTransformer

config = TransformerConfig(
    d_model=1536, num_heads=16,
    num_kv_heads=4,   # 4 KV heads shared across 16 Q heads
    use_gqa=True,
)
model = SmallTransformer(config)
```

- **Memory reduction**: ~75% smaller KV cache vs MHA (4 KV heads vs 16 Q heads)
- **Minimal accuracy degradation** compared to MHA

### Rotary Position Embeddings (RoPE)

Replaces learned/sinusoidal positional encodings with rotation-based embeddings applied inside the attention layer.

```python
config = TransformerConfig(use_rope=True)
```

- **Better generalization** to sequences longer than training length
- **No dedicated position embedding table** — saves memory

### SwiGLU Feed-Forward Network

Replaces the standard `Dense → GELU → Dense` FFN with a gated variant.

```python
config = TransformerConfig(use_swiglu=True)
```

- **~10% better parameter efficiency** vs GELU+Dense FFN
- **Improved perplexity** on language modeling benchmarks

### Flash Attention

Memory-efficient tiled attention computation.

```python
config = TransformerConfig(use_flash_attention=True)
```

- Avoids materializing the full O(n²) attention matrix
- Mathematically equivalent to standard attention

---

## Production-Grade Features

### Gradient Checkpointing

Reduces training memory by ~30-40% by recomputing activations during the backward pass instead of storing them.

```python
from src.model.transformer import TransformerConfig, SmallTransformer
from src.training.trainer import Trainer

config = TransformerConfig(gradient_checkpointing=True, ...)
# Or enable via Trainer:
trainer = Trainer(model, optimizer, train_ds, use_gradient_checkpointing=True)
```

### Mixed Precision Training

Enables FP16 or BF16 mixed precision for ~2x memory savings and speedup on modern GPUs.

```python
# Via Trainer:
trainer = Trainer(model, optimizer, train_ds, mixed_precision="fp16")

# Or globally:
from src.training.distributed import configure_mixed_precision
configure_mixed_precision("fp16")  # or "bf16" for TPU/Ampere GPUs
```

### Quantization

Reduce model size for inference deployment.

```python
from src.model.quantization import quantize_model_weights, estimate_quantized_size_gb

# Quantize all Dense weights to INT8
quant_data = quantize_model_weights(model, mode="int8")
print(f"INT8 size: {estimate_quantized_size_gb(model, mode='int8'):.2f} GB")

# INT4 (4x size reduction)
quant_data = quantize_model_weights(model, mode="int4")
print(f"INT4 size: {estimate_quantized_size_gb(model, mode='int4'):.2f} GB")
```

### Distributed Training

Auto-detect and use the best available hardware strategy.

```python
from src.training.distributed import auto_detect_strategy, configure_mixed_precision

strategy = auto_detect_strategy()   # TPU > multi-GPU > single-GPU > CPU
configure_mixed_precision("fp16")

with strategy.scope():
    model = SmallTransformer.for_size("3b")
```

For multi-GPU with explicit control:

```python
from src.training.distributed import get_distribution_strategy

strategy = get_distribution_strategy(num_gpus=4)
with strategy.scope():
    model = SmallTransformer.for_size("8b")
```

---

## Usage

### Training a 3B Model

```bash
python examples/train_3b_model.py
```

### Training from Scratch (Custom Config)

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

- **Token Embeddings**: Learned or sinusoidal positional encodings (replaced by RoPE when `use_rope=True`)
- **Transformer Blocks**: Self-attention (MHA or GQA) + FFN (standard or SwiGLU) with pre-norm
- **Attention**: Scaled dot-product attention with causal masking; optionally GQA + RoPE + Flash Attention
- **Task Heads**: Language modeling head, classification head, Q&A span head

### Supported Tasks

| Task | Model Head | Loss |
|------|-----------|------|
| `text_generation` | LM Head (vocab projection) | Cross-entropy (next token) |
| `sentiment_analysis` | Classification Head | Cross-entropy |
| `sequence_classification` | Classification Head | Cross-entropy |
| `question_answering` | Span Head (start/end logits) | Cross-entropy (span) |

### Parameter Counts

| Config | Parameters | Notes |
|--------|-----------|-------|
| Small (d=256, L=4, H=8) | ~30M | Custom baseline |
| Medium (d=512, L=6, H=8) | ~85M | Custom baseline |
| Large (d=768, L=12, H=12) | ~117M | Default config |
| XL (d=1024, L=24, H=16) | ~345M | Custom baseline |
| **1B** (d=768, L=12, GQA+RoPE+SwiGLU) | ~1.3B | Predefined `size: "1b"` |
| **3B** (d=1536, L=24, GQA+RoPE+SwiGLU) | ~3.0B | Predefined `size: "3b"` |
| **5B** (d=2048, L=32, GQA+RoPE+SwiGLU) | ~5.2B | Predefined `size: "5b"` |
| **8B** (d=2560, L=40, GQA+RoPE+SwiGLU) | ~8.0B | Predefined `size: "8b"` |

---

## Configuration

### Model Config (`config/model_config.yaml`)

```yaml
model:
  type: transformer
  # Use a predefined size (1b, 3b, 5b, 8b) or set fields manually:
  size: null
  vocab_size: 50257
  d_model: 768
  num_heads: 12
  num_kv_heads: null          # null = standard MHA; set for GQA
  num_layers: 12
  d_ff: 3072
  max_seq_length: 1024
  dropout_rate: 0.1
  task: text_generation
  # Optimizations (all off by default — fully backward compatible)
  use_gqa: false
  use_rope: false
  use_swiglu: false
  use_flash_attention: false
  gradient_checkpointing: false
  mixed_precision: null       # null, "fp16", or "bf16"
  quantization: null          # null, "int8", or "int4"
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

# Production-grade 3B model training
python examples/train_3b_model.py

# Optimized inference with quantization benchmarks
python examples/inference_optimized.py
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
pytest tests/test_model.py -v           # Model architecture tests
pytest tests/test_data.py -v            # Data pipeline tests
pytest tests/test_training.py -v        # Training utility tests
pytest tests/test_optimizations.py -v   # GQA, RoPE, SwiGLU, quantization tests
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## Data Pipeline

The repository ships a high-throughput data loading stack suitable for training on 1–100 billion token datasets.

### Supported Data Sources

| Source | Formats | Class |
|--------|---------|-------|
| Single file | `.txt`, `.jsonl`, `.parquet`, `.arrow` | `TextDataset` |
| Multiple files / directories | all of the above | `StreamingTextDataset`, `MultiFileTextDataset` |
| HuggingFace Hub | streaming or cached | `HuggingFaceLoader` |

### Streaming from Local Files

```python
from src.data.streaming_dataset import StreamingTextDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Stream from a whole directory (recursively)
dataset = StreamingTextDataset(
    paths="data/books/",
    tokenizer=tokenizer,
    max_seq_length=2048,
    recursive=True,
    shuffle=True,
)
tf_ds = dataset.get_tf_dataset(batch_size=32)
```

### Multiple Files via MultiFileTextDataset

```python
from src.data.dataset import MultiFileTextDataset

# Combine a directory and an explicit JSONL file
dataset = MultiFileTextDataset(
    paths=["data/books/", "data/web.jsonl"],
    tokenizer=tokenizer,
    max_seq_length=1024,
    recursive=True,
)
tf_ds = dataset.get_tf_dataset(batch_size=16, shuffle=True)
```

### HuggingFace Hub Integration

```python
from src.data.huggingface_loader import HuggingFaceLoader

# Stream OpenWebText without downloading the full dataset
loader = HuggingFaceLoader(
    dataset_name="openwebtext",  # short key from RECOMMENDED_DATASETS
    tokenizer=tokenizer,
    max_seq_length=1024,
    streaming=True,             # never loads full dataset into RAM
    max_samples=1_000_000,
)
tf_ds = loader.get_tf_dataset(batch_size=32)

# List available pre-training datasets
HuggingFaceLoader.list_recommended()
```

### Tokenization Caching

```python
from src.data.tokenizer_cache import TokenizerCache
from src.data.streaming_dataset import StreamingTextDataset

cache = TokenizerCache(
    cache_dir=".cache/tokenized",
    tokenizer=tokenizer,
    max_seq_length=2048,
)

if not cache.is_cached():
    src = StreamingTextDataset("data/", tokenizer, max_seq_length=2048, shuffle=False)
    cache.tokenize_texts(src.stream_texts())

tf_ds = cache.get_tf_dataset(batch_size=32, shuffle=True)
print(cache.get_stats())  # {"total_chunks": ..., "total_tokens": ..., ...}
```

### Quality Filtering & Deduplication

```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

texts = [...]
texts = preprocessor.clean_texts(texts)            # HTML, URL, e-mail removal
texts = preprocessor.filter_by_length(texts, 50, 100_000)
texts = preprocessor.deduplicate(texts)
```

### Data Statistics

```python
from src.data.statistics import DataStatistics

stats = DataStatistics(tokenizer, output_path="data/statistics.json")
stats.analyze_texts(iter(texts))
stats.save()
print(stats.get_stats()["tokens_per_text"])
```

### Data Preparation Scripts

```bash
# Prepare and split a local directory into train/val/test
python scripts/prepare_data.py local \
    --path data/raw/ --output data/prepared/ --deduplicate

# Download OpenWebText (first 100 000 documents)
python scripts/prepare_data.py huggingface \
    --dataset openwebtext \
    --max-samples 100000 \
    --output data/openwebtext_100k.jsonl

# List and download recommended large-scale datasets
python scripts/download_datasets.py --list
python scripts/download_datasets.py openwebtext slimpajama \
    --max-samples 50000 --output data/

# Compute and save data statistics
python scripts/analyze_data.py \
    --path data/train.txt \
    --output data/statistics.json
```

### Data Configuration (`config/data_config.yaml`)

```yaml
data:
  sources:
    - type: "local"
      path: "data/train.txt"
    - type: "local"
      path: "data/books/"
      recursive: true
  preprocessing:
    clean_text: true
    remove_duplicates: true
    min_length: 50
    max_length: 100000
  tokenization:
    tokenizer: "gpt2"
    max_seq_length: 2048
    cache_dir: ".cache/tokenized_data"
  loading:
    streaming: true
    batch_size: 32
    shuffle: true
```
