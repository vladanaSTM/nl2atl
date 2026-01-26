import json
from pathlib import Path

from src.evaluation.llm_judge import (
    normalize_text,
    extract_prediction_items,
    compute_metrics,
    compute_metrics_with_difficulty,
    _empty_metrics,
)
from src.evaluation.llm_judge.parser import parse_judge_response
from src.evaluation.llm_judge.prompts import format_judge_prompt
from src.evaluation.llm_judge.cache import JudgeCache
from src.evaluation.llm_judge.pipeline import LLMJudge, evaluate_prediction_file


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


def test_parse_judge_response_valid_json():
    verdict = parse_judge_response('{"correct": "yes", "reasoning": "ok"}')
    assert verdict.decision == "yes"
    assert verdict.reasoning == "ok"


def test_parse_judge_response_fallback_literal():
    verdict = parse_judge_response("output: {'correct': 'no', 'reasoning': 'bad',}")
    assert verdict.decision == "no"
    assert "bad" in verdict.reasoning


def test_format_judge_prompt_inserts_fields():
    prompt = format_judge_prompt("input text", "gold", "pred")
    assert "input text" in prompt
    assert "gold" in prompt
    assert "pred" in prompt


def test_judge_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = JudgeCache(cache_path)
    key = cache.get_cache_key("i", "g", "p", "m", "v1")
    cache.set(key, {"correct": "yes"})
    assert cache.get(key)["correct"] == "yes"


def test_evaluate_prediction_file_no_llm(tmp_path):
    prediction_path = tmp_path / "pred.json"
    payload = {
        "predictions": [
            {"input": "x", "prediction": "<<A>>F p", "gold": "<<A>>F p"},
            {"input": "y", "prediction": "<<A>>G p"},
        ]
    }
    prediction_path.write_text(json.dumps(payload))

    judge = LLMJudge(judge_model="test", no_llm=True)
    rows, stats = evaluate_prediction_file(Path(prediction_path), judge, no_llm=True)

    assert stats["auto_exact"] == 1
    assert stats["unmatched"] == 1
    assert any(r["decision_method"] == "exact" for r in rows)
