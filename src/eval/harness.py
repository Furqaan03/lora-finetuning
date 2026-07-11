"""Eval harness: score any model against a benchmark, compare base vs fine-tuned,
check for catastrophic forgetting. Model is an injected callable (prompt -> output),
so all comparison/scoring logic is testable without a GPU."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ModelFn = Callable[[str], str]


@dataclass
class BenchmarkCase:
    prompt: str
    expected: str
    category: str = "general"


@dataclass
class ModelScore:
    accuracy: float
    per_category: dict[str, float] = field(default_factory=dict)
    correct_ids: list[int] = field(default_factory=list)


def exact_match_score(model: ModelFn, benchmark: list[BenchmarkCase]) -> ModelScore:
    correct = 0
    cat_total: dict[str, int] = {}
    cat_correct: dict[str, int] = {}
    correct_ids: list[int] = []

    for i, case in enumerate(benchmark):
        cat_total[case.category] = cat_total.get(case.category, 0) + 1
        output = model(case.prompt).strip().lower()
        if case.expected.strip().lower() in output:
            correct += 1
            correct_ids.append(i)
            cat_correct[case.category] = cat_correct.get(case.category, 0) + 1

    per_category = {c: cat_correct.get(c, 0) / n for c, n in cat_total.items()}
    return ModelScore(accuracy=correct / len(benchmark) if benchmark else 0.0,
                      per_category=per_category, correct_ids=correct_ids)


@dataclass
class Comparison:
    base_accuracy: float
    finetuned_accuracy: float
    improvement: float
    helped_ids: list[int]      # fine-tuned got right, base got wrong
    regressed_ids: list[int]   # base got right, fine-tuned got wrong


def compare_models(base: ModelFn, finetuned: ModelFn, benchmark: list[BenchmarkCase]) -> Comparison:
    base_score = exact_match_score(base, benchmark)
    ft_score = exact_match_score(finetuned, benchmark)
    base_set, ft_set = set(base_score.correct_ids), set(ft_score.correct_ids)
    return Comparison(
        base_accuracy=round(base_score.accuracy, 4),
        finetuned_accuracy=round(ft_score.accuracy, 4),
        improvement=round(ft_score.accuracy - base_score.accuracy, 4),
        helped_ids=sorted(ft_set - base_set),
        regressed_ids=sorted(base_set - ft_set),
    )


def catastrophic_forgetting_check(base: ModelFn, finetuned: ModelFn,
                                  general_benchmark: list[BenchmarkCase], tolerance: float = 0.1) -> dict:
    """Fine-tuning can degrade general capabilities. Compare both models on a
    general benchmark; a drop beyond tolerance is a red flag."""
    base_acc = exact_match_score(base, general_benchmark).accuracy
    ft_acc = exact_match_score(finetuned, general_benchmark).accuracy
    drop = base_acc - ft_acc
    return {
        "base_general_accuracy": round(base_acc, 4),
        "finetuned_general_accuracy": round(ft_acc, 4),
        "drop": round(drop, 4),
        "catastrophic_forgetting": drop > tolerance,
    }
