# Fine-Tuning Pipeline with LoRA on a Domain-Specific Dataset

An end-to-end, reproducible pipeline: takes a domain-specific dataset, applies
LoRA (Low-Rank Adaptation) fine-tuning to an open-source base model, evaluates
the fine-tuned model against the base on a task-specific benchmark, and packages
the adapter for deployment — with full experiment tracking and reproducibility.

## Why this exists

Fine-tuning is where AI engineering meets ML engineering. Most candidates either
can't fine-tune at all or run a notebook once and call it done. A reproducible
pipeline with rigorous evaluation and catastrophic-forgetting checks proves you
can own the full model-customization lifecycle.

## Architecture

```
src/data/dataset.py         instruction (Alpaca) formatting, cleaning/dedup, and
                             LEAK-FREE train/val/test splits (dedup-before-split)
src/training/config.py      typed, validated config — a run is reproducible from
                             the YAML alone
src/training/train.py       QLoRA training (4-bit) with PEFT + TRL + W&B; heavy ML
                             deps are guarded so the module imports without a GPU
src/eval/harness.py         base-vs-finetuned comparison, per-category accuracy,
                             regression tracking, and catastrophic-forgetting check
src/inference/serve.py      load base + attach LoRA adapter (swap adapters without
                             reloading the base model)
configs/lora_config.yaml    all hyperparameters, with sweep ranges documented inline
```

## Design decisions

- **The test set is sacred — dedup happens BEFORE splitting.** Paraphrases of the
  same source example straddling train and test silently inflate scores. The pipeline
  deduplicates by an instruction+input fingerprint, then splits by that fingerprint,
  and `assert_no_leakage` verifies no fingerprint appears in two splits.
- **Splits are deterministic without a PRNG.** Ordering is derived from a
  seeded hash of each example's fingerprint, so the exact same split is reproducible
  across machines and runs — a requirement for honest before/after comparison.
- **QLoRA (4-bit) targets attention layers.** Rank 16 / alpha 32 / dropout 0.05 on
  `q_proj,v_proj` is the documented baseline; the config carries the sweep ranges
  (rank 8/16/32, LR 1e-4/2e-4/5e-4, epochs 1/3/5) the experiment report compares.
- **Only the adapter is saved, not the full model.** The LoRA adapter is tiny
  (<100MB) versus the 16GB+ base — so multiple domain adapters can be served off one
  base model, swapped without a reload.
- **Catastrophic forgetting is a first-class check.** Fine-tuning can wreck general
  capability; the harness runs both models on a general benchmark and flags any drop
  beyond tolerance, so "better at the task" isn't bought by "worse at everything else."
- **All non-GPU logic is tested offline.** Data prep, splits, leakage detection,
  config loading, and the entire eval/comparison harness (via injected model
  callables) are covered without torch — `requirements-core.txt` runs the full suite;
  `requirements-train.txt` is only for the actual GPU training run.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements-core.txt     # data prep + eval + tests (no GPU)
# On a CUDA machine, to actually train:
pip install -r requirements-train.txt
```

## Train (needs a GPU)

```bash
python -m src.training.train --config configs/lora_config.yaml --data data/train_examples.json
# -> logs to W&B, saves the best LoRA adapter to adapters/best
```

Without a GPU the training script exits with a clear message pointing to
`requirements-train.txt` — the data/eval/config code it shares still imports and runs.

## Evaluate

```python
from src.eval.harness import compare_models, catastrophic_forgetting_check
comp = compare_models(base_model_fn, finetuned_model_fn, benchmark)
# -> base_accuracy, finetuned_accuracy, improvement, helped_ids, regressed_ids
```

## Tests

```bash
pytest tests/ -v
```

11 tests covering data cleaning/dedup, prompt formatting, split ratios,
determinism, leakage detection, the eval harness (exact-match scoring, base-vs-
finetuned improvement, regression tracking, catastrophic-forgetting flag), and
config loading — all offline, no GPU or API key required.

## Status

Phases 1-4 complete (dataset build + leak-free splits, config-driven QLoRA
training script, eval/comparison harness with forgetting check, adapter-based
inference). The actual training run + W&B sweep require a CUDA GPU
(Colab/RunPod/local RTX) — the pipeline is built and the offline-verifiable parts
are fully tested; the trained-adapter artifacts are gitignored.
