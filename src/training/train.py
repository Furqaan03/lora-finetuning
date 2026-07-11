"""LoRA training script. Requires a GPU + torch/peft/trl (guarded imports).

Runs from a config file alone for reproducibility:
    python -m src.training.train --config configs/lora_config.yaml
"""
from __future__ import annotations

import argparse

from src.data.dataset import Example, split_dataset
from src.training.config import TrainingConfig


def load_examples(path: str) -> list[Example]:
    import json

    raw = json.loads(open(path, encoding="utf-8").read())
    return [Example(**e) for e in raw]


def train(config_path: str, data_path: str) -> None:
    """Full training run. Heavy ML deps are imported lazily so this module (and the
    config/data code it shares) imports fine on a machine without a GPU."""
    config = TrainingConfig.load(config_path)
    splits = split_dataset(load_examples(data_path))
    print(f"Loaded {len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test examples")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                   TrainingArguments)
        from trl import SFTTrainer
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            f"Training requires GPU ML deps (torch/peft/trl/transformers): {exc}\n"
            "Install with: pip install -r requirements-train.txt on a CUDA machine."
        )

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.base_model, quantization_config=bnb, device_map="auto")

    lora = LoraConfig(
        r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules, bias=config.lora.bias, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    train_ds = Dataset.from_dict({"text": [e.to_prompt() for e in splits.train]})
    val_ds = Dataset.from_dict({"text": [e.to_prompt() for e in splits.val]})

    args = TrainingArguments(
        output_dir="outputs", num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate, warmup_ratio=config.training.warmup_ratio,
        save_steps=config.training.save_steps, eval_steps=config.training.eval_steps,
        eval_strategy="steps", load_best_model_at_end=True, report_to="wandb",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         tokenizer=tokenizer, max_seq_length=config.training.max_seq_length)
    trainer.train()
    model.save_pretrained("adapters/best")   # tiny LoRA adapter (<100MB), not the full model
    print("Saved LoRA adapter to adapters/best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--data", default="data/train_examples.json")
    args = parser.parse_args()
    train(args.config, args.data)
