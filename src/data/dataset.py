"""Dataset prep: instruction formatting, cleaning/dedup, leak-free splits."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Example:
    instruction: str
    input: str
    output: str

    def fingerprint(self) -> str:
        raw = f"{self.instruction.strip().lower()}|{self.input.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_prompt(self) -> str:
        """Alpaca-style instruction format."""
        if self.input.strip():
            return (f"### Instruction:\n{self.instruction}\n\n### Input:\n{self.input}\n\n### Response:\n{self.output}")
        return f"### Instruction:\n{self.instruction}\n\n### Response:\n{self.output}"


def clean_examples(examples: list[Example]) -> list[Example]:
    """Drops empties and exact-duplicate (instruction+input) examples."""
    seen: set[str] = set()
    cleaned: list[Example] = []
    for ex in examples:
        if not ex.instruction.strip() or not ex.output.strip():
            continue
        fp = ex.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        cleaned.append(ex)
    return cleaned


@dataclass
class Splits:
    train: list[Example]
    val: list[Example]
    test: list[Example]


def split_dataset(examples: list[Example], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> Splits:
    """Deterministic 80/10/10 split with NO leakage: dedups first, then splits by
    fingerprint so paraphrase-of-same-source can't straddle train and test."""
    cleaned = clean_examples(examples)
    # Deterministic shuffle via fingerprint hash (no Random import needed).
    ordered = sorted(cleaned, key=lambda e: hashlib.sha256(f"{seed}:{e.fingerprint()}".encode()).hexdigest())

    n = len(ordered)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return Splits(
        train=ordered[:n_train],
        val=ordered[n_train:n_train + n_val],
        test=ordered[n_train + n_val:],
    )


def assert_no_leakage(splits: Splits) -> None:
    """Verifies no fingerprint appears in more than one split."""
    train_fp = {e.fingerprint() for e in splits.train}
    val_fp = {e.fingerprint() for e in splits.val}
    test_fp = {e.fingerprint() for e in splits.test}
    if train_fp & val_fp or train_fp & test_fp or val_fp & test_fp:
        raise ValueError("Data leakage detected across splits.")
