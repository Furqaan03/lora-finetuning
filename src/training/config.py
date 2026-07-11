"""Load + validate the LoRA training config (pure, testable without torch)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LoRAParams(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


class TrainingParams(BaseModel):
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    early_stopping_patience: int = 3
    save_steps: int = 50
    eval_steps: int = 50


class TrainingConfig(BaseModel):
    base_model: str
    task: str
    lora: LoRAParams = Field(default_factory=LoRAParams)
    training: TrainingParams = Field(default_factory=TrainingParams)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            base_model=data["base_model"],
            task=data["task"],
            lora=LoRAParams(**data.get("lora", {})),
            training=TrainingParams(**{k: v for k, v in data.get("training", {}).items()
                                       if k in TrainingParams.model_fields}),
        )

    def effective_batch_size(self) -> int:
        return self.training.per_device_batch_size * self.training.gradient_accumulation_steps
