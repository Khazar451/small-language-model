# Small Language Model

A production-grade language model framework in TensorFlow supporting 1B–8B parameter models, featuring advanced architecture optimizations (GQA, RoPE, SwiGLU, Flash Attention), streaming data pipelines, distributed training, and INT4/INT8 quantization. Includes a custom transformer implementation from scratch and fine-tuning support for Hugging Face pre-trained models.

## Table of Contents

- [Features](#features)
- [Model Sizes & Performance](#model-sizes--performance)
- [Architecture Optimizations](#architecture-optimizations)
- [Data Pipeline](#data-pipeline)
- [Training Infrastructure](#training-infrastructure)
- [Memory Optimization](#memory-optimization)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training from Scratch](#training-from-scratch)
  - [Fine-tuning Pre-trained Models](#fine-tuning-pre-trained-models)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Download Pre-trained Models](#download-pre-trained-models)
- [Configuration Examples](#configuration-examples)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Production-Scale Models**: Support for 1B–8B parameter transformer models
- **Custom Transformer**: GPT-style decoder-only transformer implemented from scratch in TensorFlow 2.x
- **Architecture Optimizations**: Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU activations, Flash Attention
- **Streaming Data Pipeline**: Memory-efficient streaming datasets for multi-source, multi-terabyte corpora
- **Pre-trained Model Fine-tuning**: Fine-tune GPT-2, BERT, DistilBERT, RoBERTa from Hugging Face
- **Multiple Tasks**: Text generation, sentiment analysis, sequence classification, extractive Q&A
- **Advanced Training**: Gradient checkpointing, mixed precision (fp16/bf16), distributed training (data/model/pipeline parallelism), resumable training
- **Memory Optimization**: INT4/INT8 quantization, KV-cache optimization, model sharding
- **Flexible Inference**: Top-K/Top-P sampling, beam search, greedy decoding, batch inference, optimized inference pipeline
- **GPU/TPU Support**: Automatic device detection and utilization

---

## Model Sizes & Performance

| Size | Parameters | GPU Memory (fp16) | GPU Memory (int8) | GPU Memory (int4) | Expected Perplexity |
|------|-----------|-------------------|-------------------|-------------------|---------------------|
| Small | ~117M | ~0.5 GB | ~0.25 GB | ~0.15 GB | 20–25 |
| Medium | ~345M | ~1.5 GB | ~0.75 GB | ~0.4 GB | 15–18 |
| 1B | ~1.3B | ~5 GB | ~2.5 GB | ~1.3 GB | 10–13 |
| 3B | ~3B | ~12 GB | ~6 GB | ~3 GB | 8–10 |
| 7B | ~7B | ~28 GB | ~14 GB | ~7 GB | 6–8 |

> Perplexity values are indicative and vary by dataset, tokenizer, and training duration.

---

## Architecture Optimizations

### Grouped Query Attention (GQA)

Reduces key/value head count while keeping full query heads, drastically lowering KV-cache memory at inference time.

```python
from src.model.transformer import TransformerConfig, SmallTransformer

config = TransformerConfig(
    vocab_size=50257,
    d_model=2048,
    num_heads=16,
    num_kv_heads=4,      # GQA: 4 KV heads shared across 16 query heads
    num_layers=24,
    d_ff=8192,
)
model = SmallTransformer(config)
```

### Rotary Position Embeddings (RoPE)

Position-aware attention without additive embeddings, enabling better length generalisation.

```python
config = TransformerConfig(
    positional_encoding="rope",   # Use RoPE instead of learned/sinusoidal
    rope_theta=10000.0,
)
```

### SwiGLU Activation

Replaces standard GELU in feed-forward layers for improved training stability and downstream performance.

```python
config = TransformerConfig(
    activation="swiglu",   # SwiGLU instead of gelu/relu
    d_ff=8192,
)
```

### Flash Attention

Memory-efficient attention kernel that avoids materialising the full attention matrix, enabling longer context windows.

```python
config = TransformerConfig(
    use_flash_attention=True,
    max_seq_length=8192,
)
```

---

## Data Pipeline

### Streaming Dataset

Process terabyte-scale corpora without loading everything into memory.

```python
from src.data.streaming_dataset import StreamingDataset

dataset = StreamingDataset(
    sources=["data/shard_*.jsonl"],
    tokenizer=tokenizer,
    max_seq_length=2048,
    buffer_size=10_000,
)
train_ds = dataset.get_tf_dataset(batch_size=4, shuffle=True)
```

### Multi-source Integration with HuggingFace Datasets

```python
from src.data.huggingface_loader import HuggingFaceDataLoader

loader = HuggingFaceDataLoader(
    datasets=["wikitext", "openwebtext", "c4"],
    weights=[0.3, 0.4, 0.3],   # Sampling weights
    split="train",
    streaming=True,
)
combined_ds = loader.get_interleaved_dataset(tokenizer, batch_size=8)
```

### Preprocessing Features

```python
from src.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    remove_html=True,
    normalize_whitespace=True,
    filter_min_length=50,
    filter_max_length=100_000,
    dedup=True,
)
clean_text = preprocessor.process(raw_text)
```

### Tokenizer Cache

Speed up repeated tokenisation runs by caching tokenised IDs to disk.

```python
from src.data.tokenizer_cache import TokenizerCache

cache = TokenizerCache(cache_dir=".token_cache", tokenizer=tokenizer)
token_ids = cache.encode(text)   # Returns cached result if available
```

---

## Training Infrastructure

### Gradient Checkpointing

Trade compute for memory — recompute activations during the backward pass instead of storing them.

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_ds,
    gradient_checkpointing=True,
    num_epochs=3,
)
```

### Mixed Precision Training

```python
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")  # or "mixed_bfloat16"

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_ds,
    mixed_precision=True,
)
```

### Distributed Training

```python
from src.training.distributed import DistributedTrainer

dist_trainer = DistributedTrainer(
    model=model,
    strategy="mirrored",    # "mirrored" | "tpu" | "parameter_server"
    num_gpus=8,
    train_dataset=train_ds,
    gradient_accumulation_steps=4,
)
dist_trainer.train(num_epochs=3)
```

### Resumable Training

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_ds,
    checkpoint_dir="checkpoints/my_model",
    save_steps=500,
    resume_from_checkpoint=True,   # Automatically resume from latest checkpoint
)
```

---

## Memory Optimization

### INT8 Quantization

```python
from src.model.quantization import quantize_model

quantized_model = quantize_model(
    model,
    quantization_type="int8",
    calibration_dataset=calib_ds,
)
quantized_model.save("outputs/model_int8")
```

### INT4 Quantization

```python
quantized_model = quantize_model(
    model,
    quantization_type="int4",
    group_size=128,            # Block-wise quantization group size
    calibration_dataset=calib_ds,
)
```

### Optimized Inference Pipeline

```python
from src.inference.optimized_inference import OptimizedPredictor

predictor = OptimizedPredictor(
    model_path="outputs/model_int8",
    use_kv_cache=True,
    max_batch_size=32,
    dtype="float16",
)
results = predictor.generate_batch(prompts, max_new_tokens=200)
```

---

## Repository Structure

```
small-language-model/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml          # Model architecture settings
│   ├── training_config.yaml       # Training hyperparameters
│   ├── inference_config.yaml      # Inference settings
│   └── data_config.yaml           # Data pipeline settings
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py          # Custom transformer implementation
│   │   ├── pretrained_wrapper.py   # HuggingFace model wrapper
│   │   ├── optimizations.py        # GQA, RoPE, SwiGLU, Flash Attention
│   │   └── quantization.py         # INT4/INT8 quantization
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset classes
│   │   ├── preprocessing.py        # Text preprocessing utilities
│   │   ├── streaming_dataset.py    # Streaming large-scale datasets
│   │   ├── huggingface_loader.py   # Multi-source HuggingFace integration
│   │   ├── tokenizer_cache.py      # Disk-backed tokenization cache
│   │   └── statistics.py           # Dataset statistics and analysis
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   ├── metrics.py              # Metrics tracking
│   │   └── distributed.py          # Distributed training utilities
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py            # High-level inference interface
│       ├── utils.py                # Sampling utilities
│       └── optimized_inference.py  # Optimized inference with KV-cache
├── scripts/
│   ├── train.py                    # Train from scratch
│   ├── train_3b_model.py           # End-to-end 3B model training
│   ├── finetune_pretrained.py      # Fine-tune pre-trained models
│   ├── evaluate.py                 # Model evaluation
│   ├── inference.py                # Run inference
│   ├── download_model.py           # Download HuggingFace models
│   ├── prepare_data.py             # Prepare and preprocess datasets
│   ├── download_datasets.py        # Download public datasets
│   ├── analyze_data.py             # Dataset statistics and analysis
│   └── quantize_model.py           # Post-training quantization
├── examples/
│   ├── text_generation_example.py
│   ├── qa_example.py
│   ├── sentiment_analysis_example.py
│   ├── finetuning_example.ipynb
│   ├── training_with_large_datasets.py
│   ├── inference_optimized.py
│   ├── quantization_example.py
│   └── distributed_training.py
├── deepspeed_config.json           # DeepSpeed ZeRO configuration
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_model.py               # Model architecture tests
    ├── test_data.py                # Data pipeline tests
    └── test_training.py            # Training utilities tests
```

---

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- CUDA 11.8+ (for GPU training)

### GPU Installation

```bash
pip install -r requirements.txt
pip install tensorflow[and-cuda]   # GPU support
```

### CPU-only Installation

```bash
pip install -r requirements.txt
pip install tensorflow-cpu
```

### TPU Installation

```bash
pip install -r requirements.txt
pip install tensorflow
# Configure TPU environment variables as per your cloud provider
```

### Install as a package (editable mode)

```bash
pip install -e .
```

---

## Quick Start

### End-to-end 3B Model Training

```bash
# Download and prepare training data
python scripts/download_datasets.py --dataset openwebtext --output_dir data/

# Preprocess the data
python scripts/prepare_data.py \
    --input_dir data/openwebtext \
    --output_dir data/processed \
    --tokenizer gpt2 \
    --max_seq_length 2048

# Launch 3B model training on 4 GPUs
python scripts/train_3b_model.py \
    --config config/training_config.yaml \
    --train_data data/processed/train \
    --val_data data/processed/val \
    --output_dir outputs/3b_model \
    --num_gpus 4
```

### Text Generation with GPT-2

```python
from src.model.pretrained_wrapper import PretrainedModelWrapper
from src.inference.predictor import Predictor

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

config = TransformerConfig(
    vocab_size=50257,
    d_model=2048,
    num_heads=16,
    num_kv_heads=4,
    num_layers=24,
    d_ff=8192,
    positional_encoding="rope",
    activation="swiglu",
    use_flash_attention=True,
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

## Configuration Examples

### Model Config (`config/model_config.yaml`)

```yaml
model:
  type: transformer
  vocab_size: 50257
  d_model: 2048
  num_heads: 16
  num_kv_heads: 4           # GQA: fewer KV heads for memory efficiency
  num_layers: 24
  d_ff: 8192
  max_seq_length: 2048
  dropout_rate: 0.1
  attention_dropout: 0.1
  positional_encoding: rope  # rotary position embeddings
  activation: swiglu
  use_flash_attention: true
  task: text_generation
  pretrained_model: null     # set to "gpt2", "bert-base-uncased", etc. for fine-tuning
```

### Training Config (`config/training_config.yaml`)

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8    # effective batch size = 32
  num_epochs: 3
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_grad_norm: 1.0
  optimizer:
    type: adamw
    betas: [0.9, 0.95]
    epsilon: 1.0e-8
  lr_scheduler: cosine
  mixed_precision: bf16
  gradient_checkpointing: true
  distributed:
    strategy: mirrored
    num_gpus: 8
  checkpointing:
    save_steps: 1000
    save_total_limit: 5
    resume_from_checkpoint: true
  early_stopping:
    enabled: true
    patience: 3
  logging_steps: 50
  eval_steps: 500
  seed: 42
```

### Data Config (`config/data_config.yaml`)

```yaml
data:
  sources:
    - path: data/openwebtext
      weight: 0.4
      format: jsonl
    - path: data/wikipedia
      weight: 0.3
      format: jsonl
    - path: data/books
      weight: 0.3
      format: txt
  tokenizer: gpt2
  max_seq_length: 2048
  streaming: true
  buffer_size: 50000
  preprocessing:
    remove_html: true
    normalize_whitespace: true
    filter_min_length: 50
    dedup: true
  tokenizer_cache:
    enabled: true
    cache_dir: .token_cache
```

---

## Examples

### Training with Large Datasets

```bash
python examples/training_with_large_datasets.py \
    --config config/training_config.yaml \
    --data_config config/data_config.yaml \
    --output_dir outputs/large_run
```

```python
# examples/training_with_large_datasets.py
from src.data.streaming_dataset import StreamingDataset
from src.data.huggingface_loader import HuggingFaceDataLoader
from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.trainer import Trainer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

loader = HuggingFaceDataLoader(
    datasets=["wikitext", "openwebtext"],
    weights=[0.4, 0.6],
    streaming=True,
)
train_ds = loader.get_interleaved_dataset(tokenizer, batch_size=4)

config = TransformerConfig(
    d_model=2048, num_heads=16, num_kv_heads=4,
    num_layers=24, d_ff=8192,
    positional_encoding="rope", activation="swiglu",
    use_flash_attention=True,
)
model = SmallTransformer(config)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    gradient_checkpointing=True,
    mixed_precision=True,
    checkpoint_dir="checkpoints/large_run",
    resume_from_checkpoint=True,
)
trainer.train(num_epochs=3)
```

### Optimized Inference

```bash
python examples/inference_optimized.py \
    --model_path outputs/model_int8 \
    --prompts_file data/prompts.txt \
    --output_file results/generated.txt
```

```python
# examples/inference_optimized.py
from src.model.quantization import quantize_model
from src.inference.optimized_inference import OptimizedPredictor

# Quantize to INT8
quantized = quantize_model(model, quantization_type="int8")
quantized.save("outputs/model_int8")

# Run optimized inference
predictor = OptimizedPredictor(
    model_path="outputs/model_int8",
    use_kv_cache=True,
    dtype="float16",
)
outputs = predictor.generate_batch(
    prompts=["The future of AI is", "Large language models can"],
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
)
for text in outputs:
    print(text)
```

### Distributed Training

```bash
# 8-GPU data-parallel training
python examples/distributed_training.py \
    --config config/training_config.yaml \
    --num_gpus 8 \
    --strategy mirrored

# Multi-node training via DeepSpeed
deepspeed examples/distributed_training.py \
    --deepspeed deepspeed_config.json \
    --config config/training_config.yaml
```

```python
# examples/distributed_training.py
from src.training.distributed import DistributedTrainer

dist_trainer = DistributedTrainer(
    model=model,
    strategy="mirrored",
    num_gpus=8,
    train_dataset=train_ds,
    gradient_accumulation_steps=4,
)
dist_trainer.train(num_epochs=3)
```

### Quantization

```bash
python examples/quantization_example.py \
    --model_path outputs/my_model \
    --quantization int4 \
    --output_path outputs/my_model_int4
```

```python
# examples/quantization_example.py
from src.model.quantization import quantize_model

# INT4 quantization
int4_model = quantize_model(
    model,
    quantization_type="int4",
    group_size=128,
    calibration_dataset=calib_ds,
)
int4_model.save("outputs/my_model_int4")
print(f"INT4 model size: {int4_model.get_size_gb():.2f} GB")
```

Run the classic example scripts:

```bash
python examples/text_generation_example.py
python examples/sentiment_analysis_example.py
python examples/qa_example.py
jupyter notebook examples/finetuning_example.ipynb
```

---

## Benchmarks

### Training Throughput (tokens/second)

| Model Size | 1× A100 (fp16) | 4× A100 (fp16) | 8× A100 (bf16) | 8× A100 + GC* |
|-----------|----------------|----------------|-----------------|---------------|
| 117M | ~120,000 | ~460,000 | ~900,000 | ~650,000 |
| 1.3B | ~18,000 | ~68,000 | ~130,000 | ~95,000 |
| 3B | ~7,000 | ~26,000 | ~50,000 | ~36,000 |
| 7B | ~2,800 | ~10,500 | ~20,000 | ~14,000 |

*GC = gradient checkpointing enabled. Reduces throughput ~30% but halves peak memory usage.

### Inference Latency (ms, batch size 1, 200 new tokens)

| Model Size | fp32 | fp16 | int8 | int4 |
|-----------|------|------|------|------|
| 117M | 85 | 45 | 28 | 18 |
| 1.3B | 620 | 320 | 195 | 120 |
| 3B | 1,450 | 740 | 450 | 275 |
| 7B | 3,400 | 1,750 | 1,050 | 640 |

> Benchmarks run on NVIDIA A100 80GB with CUDA 11.8, TensorFlow 2.13.

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

---

## Troubleshooting

### Out-of-Memory (OOM) Errors

**Symptom**: `ResourceExhaustedError: OOM when allocating tensor`

**Solutions**:
1. Reduce `batch_size` in `config/training_config.yaml`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable gradient checkpointing: set `gradient_checkpointing: true`
4. Switch to a lower-precision dtype: `mixed_precision: bf16`
5. Quantize the model to INT8 or INT4 before inference

### Slow Training

**Symptom**: Training throughput much lower than benchmark values.

**Solutions**:
1. Verify GPU utilisation with `nvidia-smi`; if < 80%, increase `batch_size`
2. Enable mixed precision training (`bf16` preferred on Ampere GPUs)
3. Enable `use_flash_attention: true` in the model config
4. Set `pin_memory: true` and use multiple data-loader workers

### Training Loss Not Decreasing

**Solutions**:
1. Check that the learning rate is not too high — try `1e-4` to `3e-4` for large models
2. Ensure warmup steps are proportional to dataset size (typically 1–2% of total steps)
3. Verify the tokenizer `pad_token` is correctly set (`tokenizer.pad_token = tokenizer.eos_token`)
4. Inspect data preprocessing — filter very short or very long sequences

### Checkpoint Loading Fails

**Symptom**: Errors when loading saved weights.

**Solutions**:
1. Ensure the model config matches the one used during training
2. Use `.weights.h5` extension for Keras 3 weight files: `model.save_weights("path.weights.h5")`
3. If resuming training, set `resume_from_checkpoint: true` in the training config

### Distributed Training Hangs

**Solutions**:
1. Verify all GPUs are visible: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
2. Check NCCL version compatibility with your CUDA installation
3. Reduce `num_gpus` and test with `strategy: mirrored` before moving to multi-node

---

## Performance Tips

### Training

- **Effective batch size**: Use `gradient_accumulation_steps` to reach an effective batch size of 256–2048 tokens per step without increasing per-GPU memory.
- **Learning rate scaling**: Scale the learning rate linearly with the effective batch size (e.g., double the batch size → double the LR).
- **BF16 over FP16**: On Ampere GPUs (A100, RTX 30xx), prefer `bf16` — it has the same dynamic range as `fp32` and avoids gradient overflow.
- **Flash Attention**: Always enable `use_flash_attention: true` for sequences longer than 512 tokens.
- **GQA**: For models ≥ 3B parameters, use `num_kv_heads = num_heads // 4` to reduce KV-cache memory by 4×.

### Inference

- **KV-cache**: Enable `use_kv_cache=True` in `OptimizedPredictor` for autoregressive generation — avoids redundant attention computation.
- **Batch generation**: Batch multiple prompts together (`generate_batch`) for significantly higher GPU utilisation than sequential generation.
- **Quantization**: INT8 quantization reduces model size by ~50% with minimal quality loss. INT4 reduces by ~75% with slightly more degradation.
- **Sequence length**: Set `max_new_tokens` conservatively — longer sequences increase latency quadratically without Flash Attention.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and create a feature branch
2. Write tests for any new functionality
3. Ensure all existing tests pass: `pytest tests/ -v`
4. Format code with `black` and lint with `flake8`
5. Open a pull request with a clear description of the changes

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{small_language_model,
  title  = {Small Language Model: A Production-Grade Transformer Framework},
  year   = {2024},
  url    = {https://github.com/Khazar451/small-language-model},
}
```
