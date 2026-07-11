from src.eval.harness import (BenchmarkCase, catastrophic_forgetting_check, compare_models,
                              exact_match_score)
from src.training.config import TrainingConfig

BENCH = [
    BenchmarkCase("classify: terminate on notice", "termination", "clauses"),
    BenchmarkCase("classify: hold in confidence", "confidentiality", "clauses"),
    BenchmarkCase("classify: governed by Delaware", "governing_law", "clauses"),
]


def test_exact_match_scoring():
    perfect = lambda p: {"terminate on notice": "termination", "hold in confidence": "confidentiality",
                         "governed by Delaware": "governing_law"}[p.split(": ")[1]]
    score = exact_match_score(perfect, BENCH)
    assert score.accuracy == 1.0


def test_compare_models_improvement():
    base = lambda p: "unknown"                    # base gets everything wrong
    finetuned = lambda p: "termination" if "terminate" in p else "governing_law" if "Delaware" in p else "confidentiality"
    comp = compare_models(base, finetuned, BENCH)
    assert comp.finetuned_accuracy > comp.base_accuracy
    assert comp.improvement > 0
    assert len(comp.helped_ids) == 3


def test_regression_tracking():
    # base right on case 0, fine-tuned wrong on it -> regression
    base = lambda p: "termination" if "terminate" in p else "confidentiality" if "confidence" in p else "governing_law"
    finetuned = lambda p: "wrong" if "terminate" in p else "confidentiality" if "confidence" in p else "governing_law"
    comp = compare_models(base, finetuned, BENCH)
    assert 0 in comp.regressed_ids


def test_catastrophic_forgetting_flagged():
    general = [BenchmarkCase("2+2", "4"), BenchmarkCase("capital of France", "paris")]
    base = lambda p: "4" if "2+2" in p else "paris"
    forgot = lambda p: "nonsense"
    result = catastrophic_forgetting_check(base, forgot, general, tolerance=0.1)
    assert result["catastrophic_forgetting"] is True


def test_config_loads_from_yaml():
    config = TrainingConfig.load("configs/lora_config.yaml")
    assert config.lora.r == 16
    assert config.lora.alpha == 32
    assert config.effective_batch_size() == 16  # 4 * 4
