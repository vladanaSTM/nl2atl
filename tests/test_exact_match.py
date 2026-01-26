import pytest

from src.evaluation.exact_match import ExactMatchEvaluator


def test_clean_output_strips_think_and_tags():
    evaluator = ExactMatchEvaluator()
    response = "<|im_start|>assistant\n<think>reason</think>\n<<A>>F p<|im_end|>"
    cleaned = evaluator.clean_output(response, model_type="qwen")
    assert cleaned == "<<A>>F p"


def test_normalize_symbols_equivalence():
    evaluator = ExactMatchEvaluator()
    a = "<<A>>G (p ∧ q)"
    b = "<<A>>G (p && q)"
    assert evaluator.normalize(a) == evaluator.normalize(b)


def test_evaluate_predictions_length_mismatch():
    evaluator = ExactMatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate_predictions([{}, {}, {}], [{}])


def test_evaluate_single_preserves_latency():
    evaluator = ExactMatchEvaluator()
    prediction = {"input": "x", "generated": "<<A>>F p", "latency_ms": 12.3}
    reference = {"input": "x", "output": "<<A>>F p"}
    result = evaluator.evaluate_single(prediction, reference)
    assert result["latency_ms"] == 12.3
