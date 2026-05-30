import pytest

from src.evaluation.exact_match import ExactMatchEvaluator


def test_clean_output_strips_think_and_tags():
    evaluator = ExactMatchEvaluator()
    response = "<|im_start|>assistant\n<think>reason</think>\n<<A>>F p<|im_end|>"
    cleaned = evaluator.clean_output(response, model_type="qwen")
    assert cleaned == "<<A>>F p"


def test_clean_output_extracts_mistral_inst_answer():
    evaluator = ExactMatchEvaluator()
    response = "<s>[INST] Convert to ATL formula: <<Bad>>F prompt [/INST] <<A>>F p</s>"
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == "<<A>>F p"


def test_clean_output_extracts_phi_answer():
    evaluator = ExactMatchEvaluator()
    response = "<|assistant|>\nFinal formula: <<A>>G(p -> q)<|end|>"
    cleaned = evaluator.clean_output(response, model_type="phi3")
    assert cleaned == "<<A>>G(p -> q)"


def test_clean_output_extracts_generic_code_fence_answer():
    evaluator = ExactMatchEvaluator()
    response = "Assistant:\n```atl\n<<A,B>>X(p && q)\n```"
    cleaned = evaluator.clean_output(response, model_type="generic")
    assert cleaned == "<<A,B>>X(p && q)"


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


def test_evaluate_single_accepts_any_expected_option():
    evaluator = ExactMatchEvaluator()
    prediction = {"input": "x", "generated": "<<A,B>>X p"}
    reference = {
        "input": "x",
        "output_1": "<<A>>X p_1 && <<B>>X p_2",
        "output_2": "<<A,B>>X p",
    }

    result = evaluator.evaluate_single(prediction, reference)

    assert result["exact_match"] == 1
    assert result["expected_options"] == [
        "<<A,B>>X p",
        "<<A>>X p_1 && <<B>>X p_2",
    ]
