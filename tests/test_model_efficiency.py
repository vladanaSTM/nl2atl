import json
from pathlib import Path

from src.evaluation.model_efficiency import (
    build_efficiency_report,
    resolve_prediction_files,
)


def test_latency_derivations(tmp_path: Path) -> None:
    prediction_path = tmp_path / "model_run.json"

    metadata = {
        "run_id": "test-run",
        "model": "local-a100",
        "model_short": "a100",
        "total_samples": 2,
        "duration_seconds": 3600.0,
        "metrics": {
            "exact_match": 0.5,
        },
    }

    predictions = [
        {"id": "ex1"},
        {"id": "ex2"},
    ]

    prediction_path.write_text(
        json.dumps({"metadata": metadata, "predictions": predictions})
    )

    report = build_efficiency_report([prediction_path])
    assert report["totals"]["models"] == 1

    entry = report["models"][0]
    assert entry["accuracy"] == 0.5
    assert entry["latency_total_ms"] == 3600000.0
    assert entry["latency_mean_ms"] == 1800000.0
    assert entry["duration_seconds"] == 3600.0
    assert entry["throughput_samples_per_sec"] == 0.000556
    assert entry["accuracy_per_second"] == 0.000278


def test_resolve_prediction_files(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    p1 = predictions_dir / "run1.json"
    p1.write_text("{}")
    p2 = predictions_dir / "run2.json"
    p2.write_text("{}")

    resolved_all = resolve_prediction_files(predictions_dir, ["all"])
    assert p1 in resolved_all and p2 in resolved_all

    resolved_named = resolve_prediction_files(predictions_dir, ["run1"])
    assert resolved_named == [p1]


def test_build_efficiency_report_with_summary(tmp_path):
    prediction_path = tmp_path / "model_run.json"
    prediction_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "run_id": "run-1",
                    "model": "m1",
                    "model_short": "m1",
                    "total_samples": 2,
                    "latency_mean_ms": 100.0,
                },
                "predictions": [
                    {"exact_match": True, "latency_ms": 50},
                    {"exact_match": False, "latency_ms": 150},
                ],
            }
        )
    )

    summary = {
        "per_file": [
            {
                "source_file": prediction_path.name,
                "metrics": {"accuracy": 0.8},
            }
        ]
    }

    report = build_efficiency_report([prediction_path], judge_summary=summary)
    entry = report["models"][0]
    assert entry["accuracy"] == 0.8
    assert entry["accuracy_source"] == "llm_judge"
    assert entry["efficiency_score"] is not None
    assert report["totals"]["with_latency"] == 1
    assert "pareto_frontier" in report
