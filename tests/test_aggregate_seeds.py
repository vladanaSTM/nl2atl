import json

from src.cli.aggregate_seeds import aggregate_predictions, _build_notebook


def _write_evaluated(path, *, judge_model, seed, accuracy):
    payload = {
        "judge_model": judge_model,
        "model": "Qwen/Qwen3-3B",
        "model_short": "qwen-3b",
        "condition": "baseline_zero_shot",
        "finetuned": False,
        "few_shot": False,
        "seed": seed,
        "detailed_results": [
            {
                "input": "x",
                "prediction": "<<A>>F p",
                "gold_options": ["<<A>>F p"],
                "correct": "yes" if accuracy == 1.0 else "no",
                "decision_method": "exact" if accuracy == 1.0 else "llm",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_predictions_separates_judges_by_default(tmp_path):
    evaluated_dir = tmp_path / "evaluated_datasets"
    _write_evaluated(
        evaluated_dir / "judge-a" / "qwen_seed42__judge-judge-a.json",
        judge_model="judge-a",
        seed=42,
        accuracy=1.0,
    )
    _write_evaluated(
        evaluated_dir / "judge-b" / "qwen_seed42__judge-judge-b.json",
        judge_model="judge-b",
        seed=42,
        accuracy=0.0,
    )

    aggregates = aggregate_predictions(evaluated_dir)

    assert len(aggregates) == 2
    assert {entry["judge_model"] for entry in aggregates} == {"judge-a", "judge-b"}
    assert all(entry["num_seeds"] == 1 for entry in aggregates)
    assert all(entry["num_runs"] == 1 for entry in aggregates)
    assert all("ci95" in entry["metrics"]["accuracy"] for entry in aggregates)


def test_aggregate_predictions_can_explicitly_combine_judges(tmp_path):
    evaluated_dir = tmp_path / "evaluated_datasets"
    _write_evaluated(
        evaluated_dir / "judge-a" / "qwen_seed42__judge-judge-a.json",
        judge_model="judge-a",
        seed=42,
        accuracy=1.0,
    )
    _write_evaluated(
        evaluated_dir / "judge-b" / "qwen_seed42__judge-judge-b.json",
        judge_model="judge-b",
        seed=42,
        accuracy=0.0,
    )

    aggregates = aggregate_predictions(evaluated_dir, combine_judges=True)

    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate["judge_model"] is None
    assert aggregate["judge_models"] == ["judge-a", "judge-b"]
    assert aggregate["num_seeds"] == 1
    assert aggregate["num_runs"] == 2


def test_aggregate_notebook_cells_have_language_metadata(tmp_path):
    nb = _build_notebook([], str(tmp_path / "aggregate.json"))

    assert nb["cells"]
    assert all(cell["metadata"].get("language") for cell in nb["cells"])
