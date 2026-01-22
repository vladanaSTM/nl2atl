from src.llm_judge import (
    normalize_text,
    extract_prediction_items,
    compute_metrics,
    compute_metrics_with_difficulty,
    _empty_metrics,
)


def test_normalize_text():
    assert normalize_text(None) == ""
    assert normalize_text("  a   b \n c ") == "a b c"


def test_extract_prediction_items():
    data = [{"input": "x", "generated": "g", "expected": "e", "exact_match": 1}]
    parsed = extract_prediction_items(data)
    assert parsed and parsed[0]["input"] == "x"


def test_extract_prediction_items_with_alt_fields():
    data = {
        "detailed_results": [
            {"input": "x", "prediction": "p", "gold": "g"},
            {"input": "y", "model_output": "p2", "reference": "g2"},
        ]
    }
    parsed = extract_prediction_items(data)
    assert len(parsed) == 2
    assert parsed[0]["prediction"] == "p"
    assert parsed[1]["gold"] == "g2"


def test_compute_metrics_basic():
    rows = [
        {"decision_method": "exact", "correct": "yes"},
        {"decision_method": "llm", "correct": "no"},
        {"decision_method": "llm", "correct": "yes"},
    ]
    metrics = compute_metrics(rows)
    assert "accuracy" in metrics
    # total evaluated should be 3
    assert metrics["total_evaluated"] == 3
    assert metrics["correct"] == 2


def test_compute_metrics_with_difficulty():
    rows = [
        {"decision_method": "exact", "correct": "yes", "difficulty": "easy"},
        {"decision_method": "llm", "correct": "no", "difficulty": "hard"},
        {"decision_method": "llm", "correct": "yes", "difficulty": "easy"},
    ]
    metrics = compute_metrics_with_difficulty(rows)
    assert "by_difficulty" in metrics
    assert metrics["by_difficulty"]["easy"]["accuracy"] == 1.0


def test_empty_metrics():
    assert _empty_metrics()["total_evaluated"] == 0
