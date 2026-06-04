import ast
import json

from src.evaluation.publication_notebook import (
    DEFAULT_PUBLICATION_NOTEBOOK_NAME,
    build_publication_notebook,
)


def test_publication_notebook_cells_have_language_metadata_and_focused_content(
    tmp_path,
):
    eval_dir = tmp_path / "LLM-evaluation"
    eval_dir.mkdir()

    (eval_dir / "seed_aggregate_metrics_from_judged.json").write_text(
        json.dumps(
            [
                {
                    "model_short": "qwen-3b",
                    "condition": "baseline_zero_shot",
                    "judge_model": "judge-a",
                    "num_seeds": 3,
                    "num_runs": 3,
                    "metrics": {
                        "accuracy": {"mean": 0.72, "std": 0.02, "ci95": 0.03},
                        "exact_match_rate": {"mean": 0.4},
                        "llm_approval_rate": {"mean": 0.5},
                        "accuracy_boost_from_llm": {"mean": 0.32},
                        "latency_mean_ms": {"mean": 120.0},
                    },
                    "judge_agreement": {
                        "confidence_score": {"mean": 0.78},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (eval_dir / "efficiency_report.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_short": "qwen-3b",
                        "condition": "baseline_zero_shot",
                        "judge_model": "judge-a",
                        "num_seeds": 3,
                        "num_runs": 3,
                        "accuracy": 0.72,
                        "latency_mean_ms": 120.0,
                    }
                ],
                "pareto_frontier": [
                    {
                        "model": "qwen-3b",
                        "condition": "baseline_zero_shot",
                        "judge_model": "judge-a",
                        "accuracy": 0.72,
                        "latency_mean_ms": 120.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "agreement_report.json").write_text(
        json.dumps(
            {
                "judges": ["judge-a", "judge-b"],
                "items_with_multiple_judges": 10,
                "summary": {"mean_pairwise_kappa": 0.7},
                "krippendorff_alpha": {"alpha": 0.68},
                "pairwise_cohen_kappa": {
                    "judge-a_vs_judge-b": {"kappa": 0.7, "n_common": 10}
                },
                "per_source_file": {},
                "disagreements": {"total_count": 1, "sample": []},
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "summary__judge-judge-a.json").write_text(
        json.dumps(
            {
                "judge_model": "judge-a",
                "prompt_version": "v1",
                "overall": {
                    "total_evaluated": 10,
                    "accuracy": 0.72,
                    "correct": 7,
                    "exact_match": {"rate": 0.4},
                    "llm_judged": {"rate": 0.6, "approval_rate": 0.5},
                    "accuracy_boost_from_llm": 0.32,
                },
                "totals": {"auto_exact": 4, "llm_calls": 6, "no_llm": 0},
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "reproducibility_manifest.json").write_text(
        json.dumps({"reports": [], "inputs": {}}), encoding="utf-8"
    )

    notebook_path = eval_dir / DEFAULT_PUBLICATION_NOTEBOOK_NAME
    build_publication_notebook(eval_dir, notebook_path)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["cells"]
    assert all(cell["metadata"].get("language") for cell in notebook["cells"])

    joined_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "Main Model Results" in joined_source
    assert "Combined-Judge Ranking" in joined_source
    assert "Paired Top-Model Tests" in joined_source
    assert "deterministic_exact_match_lower_bound" in joined_source
    assert "randomization_p_value" in joined_source
    assert "Judge Reliability" in joined_source
    assert "Accuracy-Latency Tradeoff" in joined_source
    assert "efficiency_score" not in joined_source

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
