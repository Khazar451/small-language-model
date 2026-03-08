"""
Example: Distributed training across multiple GPUs.

Run (data-parallel, 8 GPUs):
    python examples/distributed_training.py \\
        --config config/training_config.yaml \\
        --num_gpus 8 \\
        --strategy mirrored

Run (multi-node via DeepSpeed — requires deepspeed installed):
    deepspeed examples/distributed_training.py \\
        --deepspeed deepspeed_config.json \\
        --config config/training_config.yaml
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from transformers import AutoTokenizer

import tensorflow as tf
from src.data.streaming_dataset import StreamingDataset
from src.model.transformer import SmallTransformer, TransformerConfig
from src.training.distributed import DistributedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--train_data", default="data/train.txt")
    parser.add_argument("--output_dir", default="outputs/distributed")
    parser.add_argument("--num_gpus", type=int, default=0, help="0 = all available")
    parser.add_argument("--strategy", default="mirrored",
                        choices=["mirrored", "tpu", "parameter_server", "one_device"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        train_cfg = yaml.safe_load(fh).get("training", {})

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = StreamingDataset(
        sources=[args.train_data],
        tokenizer=tokenizer,
        max_seq_length=2048,
    ).get_tf_dataset(batch_size=train_cfg.get("batch_size", 4), shuffle=True)

    def model_fn():
        cfg = TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_kv_heads=4,
            num_layers=24,
            d_ff=4096,
            positional_encoding="rope",
            activation="swiglu",
        )
        return SmallTransformer(cfg)

    dist_trainer = DistributedTrainer(
        model_fn=model_fn,
        strategy=args.strategy,
        num_gpus=args.num_gpus,
        train_dataset=train_ds,
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
        ),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        output_dir=args.output_dir,
        num_epochs=train_cfg.get("num_epochs", 3),
    )

    logger.info(
        "Distributed training with %d replica(s), strategy=%s",
        dist_trainer.num_replicas,
        args.strategy,
    )
    dist_trainer.train()


if __name__ == "__main__":
    main()
