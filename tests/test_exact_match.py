import pytest

from src.evaluation.exact_match import ExactMatchEvaluator
from src.infra.azure import GenerationResult


def test_clean_output_keeps_answer_between_qwen_tokens():
    evaluator = ExactMatchEvaluator()
    response = "<|im_start|>assistant\n<think>reason</think>\n<<A>>F p<|im_end|>"
    cleaned = evaluator.clean_output(response, model_type="qwen")
    assert cleaned == "<think>reason</think>\n<<A>>F p"


def test_clean_output_extracts_mistral_inst_answer():
    evaluator = ExactMatchEvaluator()
    response = "<s>[INST] Convert to ATL formula: <<Bad>>F prompt [/INST] <<A>>F p</s>"
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == "<<A>>F p"


def test_clean_output_preserves_mistral_explanation_suffix():
    evaluator = ExactMatchEvaluator()
    response = (
        "<<ClinicalKiosk1>>G(consentPrompt -> next consented), "
        "where <<ClinicalKiosk1>> represents the clinical kiosk"
    )
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == response


def test_clean_output_preserves_mistral_alternatives():
    evaluator = ExactMatchEvaluator()
    response = (
        "<<RecoverySystem,StorageManager>>G(next -> backup_triggered), "
        "or equivalently <<RecoverySystem>>G(next -> backup_triggered) "
        "&& <<StorageManager>>G(next -> backup_triggered)"
    )
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == response


def test_clean_output_preserves_valid_coalition_and_conjunction_commas():
    evaluator = ExactMatchEvaluator()
    response = (
        "<<Paramedic,HospitalDispatcher>>G stretcher_path_clear "
        "&& <<AuditService>>F alarm_record_archived, where the second clause is shared"
    )
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == response


def test_clean_output_extracts_phi_answer():
    evaluator = ExactMatchEvaluator()
    response = "<|assistant|>\nFinal formula: <<A>>G(p -> q)<|end|>"
    cleaned = evaluator.clean_output(response, model_type="phi3")
    assert cleaned == "Final formula: <<A>>G(p -> q)"


def test_clean_output_preserves_phi_repeated_token_collapse():
    evaluator = ExactMatchEvaluator()
    response = "<|assistant|>\ncancer cancer cancer cancer cancer cancer cancer cancer cancer<|end|>"
    cleaned = evaluator.clean_output(response, model_type="phi3")
    assert cleaned == "cancer cancer cancer cancer cancer cancer cancer cancer cancer"


def test_clean_output_preserves_concatenated_repetition():
    evaluator = ExactMatchEvaluator()
    response = "<|assistant|>\nwhowhowhowhowhowhowhowhowhowhowho<|end|>"
    cleaned = evaluator.clean_output(response, model_type="phi3")
    assert cleaned == "whowhowhowhowhowhowhowhowhowhowho"


def test_clean_output_preserves_incomplete_mistral_coalition():
    evaluator = ExactMatchEvaluator()
    response = "<<Monitor1,Monitor2,Monitor3,Monitor4,G(payload_safe)"
    cleaned = evaluator.clean_output(response, model_type="mistral")
    assert cleaned == response


def test_clean_output_extracts_generic_code_fence_answer():
    evaluator = ExactMatchEvaluator()
    response = "Assistant:\n```atl\n<<A,B>>X(p && q)\n```"
    cleaned = evaluator.clean_output(response, model_type="generic")
    assert cleaned == "```atl\n<<A,B>>X(p && q)\n```"


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
    reference = {"input": "x", "outputs": ["<<A>>F p"]}
    result = evaluator.evaluate_single(prediction, reference)
    assert result["latency_ms"] == 12.3


def test_evaluate_single_requires_all_expected_options():
    evaluator = ExactMatchEvaluator()
    reference = {
        "input": "x",
        "outputs": [
            "<<A>>X p_1 && <<B>>X p_2",
            "<<A,B>>X p",
        ],
    }

    # A prediction with only one required reading is not an exact match.
    partial = evaluator.evaluate_single(
        {"input": "x", "generated": "<<A,B>>X p"}, reference
    )
    assert partial["exact_match"] == 0

    # A prediction with all required readings (any order) is an exact match.
    complete = evaluator.evaluate_single(
        {"input": "x", "generated": "<<A,B>>X p\n<<A>>X p_1 && <<B>>X p_2"},
        reference,
    )
    assert complete["exact_match"] == 1
    assert complete["expected_options"] == [
        "<<A>>X p_1 && <<B>>X p_2",
        "<<A,B>>X p",
    ]


def test_aggregate_metrics_reports_exact_match_only():
    evaluator = ExactMatchEvaluator()
    metrics = evaluator.evaluate_predictions(
        [{"input": "x", "generated": ""}],
        [{"input": "x", "outputs": ["<<A>>F p"]}],
    )

    assert metrics == {"n_examples": 1, "exact_match": 0.0}


def test_evaluate_model_records_generation_provenance(monkeypatch):
    def fake_generate(model, tokenizer, prompt, max_new_tokens=256, return_usage=False):
        assert return_usage is True
        assert "Convert to ATL formula: x" in prompt
        return GenerationResult(
            text="<<A>>F p",
            usage={
                "tokens_input": 10,
                "tokens_output": 4,
                "tokens_total": 14,
            },
        )

    monkeypatch.setattr("src.models.registry.generate", fake_generate)

    evaluator = ExactMatchEvaluator()
    metrics = evaluator.evaluate_model(
        model=object(),
        tokenizer=None,
        test_data=[{"id": "row-1", "input": "x", "outputs": ["<<A>>F p"]}],
        model_type="generic",
        few_shot=True,
        num_few_shot=2,
        verbose=False,
    )

    row = evaluator.results[0]
    assert metrics["exact_match"] == 1.0
    assert row["generated"] == "<<A>>F p"
    # The raw generation equals the cleaned output here, so it is omitted to
    # avoid duplicating "generated".
    assert "raw_generation" not in row
    assert row["generation_prompt_sha256"]
    assert row["generation_config"]["do_sample"] is False
    assert row["token_usage"]["tokens_total"] == 14
    assert len(row["few_shot_example_ids"]) == 2
    assert "generation_prompt" not in row
    assert "few_shot_examples" not in row


def test_evaluate_model_keeps_raw_generation_when_cleaning_changes_it(monkeypatch):
    def fake_generate(model, tokenizer, prompt, max_new_tokens=256, return_usage=False):
        # Emulate a local model emitting chat/stop tokens around the answer.
        return GenerationResult(
            text="<<A>>F p<|im_end|>",
            usage={
                "tokens_input": 10,
                "tokens_output": 5,
                "tokens_total": 15,
            },
        )

    monkeypatch.setattr("src.models.registry.generate", fake_generate)

    evaluator = ExactMatchEvaluator()
    evaluator.evaluate_model(
        model=object(),
        tokenizer=None,
        test_data=[{"id": "row-1", "input": "x", "outputs": ["<<A>>F p"]}],
        model_type="qwen",
        few_shot=False,
        verbose=False,
    )

    row = evaluator.results[0]
    assert row["generated"] == "<<A>>F p"
    # Cleaning stripped the stop token, so the pre-clean text is preserved.
    assert row["raw_generation"] == "<<A>>F p<|im_end|>"
