"""Inference serving: load base model + attach LoRA adapter, expose A/B endpoint.

Heavy deps are guarded so this module imports without a GPU; the prompt
formatting it shares with training/eval is pure."""
from __future__ import annotations

from src.data.dataset import Example


def format_inference_prompt(instruction: str, input_text: str = "") -> str:
    """Same Alpaca format as training — must match or the adapter underperforms."""
    return Example(instruction=instruction, input=input_text, output="").to_prompt().replace("### Response:\n", "### Response:\n").rstrip()


def load_model_with_adapter(base_model: str, adapter_path: str):
    """Loads the base model once and attaches the tiny LoRA adapter — adapters can
    be swapped without reloading the (large) base model."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"Inference requires torch/peft/transformers: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer
