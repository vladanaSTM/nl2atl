import json
from pathlib import Path

from src.evaluation.model_efficiency import build_efficiency_report


def test_gpu_hour_cost_derivation(tmp_path: Path) -> None:
    prediction_path = tmp_path / "model_run.json"

    metadata = {
        "run_id": "test-run",
        "model": "local-a100",
        "model_short": "a100",
        "total_samples": 2,
        "gpu_hour_usd": 10.0,
        "duration_seconds": 3600.0,
        "metrics": {
            "total_tokens_input": 400,
            "total_tokens_output": 600,
            "total_tokens": 1000,
        },
    }

    predictions = [
        {"id": "ex1", "tokens_input": 200, "tokens_output": 300},
        {"id": "ex2", "tokens_input": 200, "tokens_output": 300},
    ]

    prediction_path.write_text(
        json.dumps({"metadata": metadata, "predictions": predictions})
    )

    report = build_efficiency_report([prediction_path])
    assert report["totals"]["models"] == 1

    entry = report["models"][0]
    assert entry["gpu_hour_usd"] == 10.0
    assert entry["cost_total_usd"] == 10.0
    assert entry["avg_cost_usd"] == 5.0
    assert entry["tokens_per_hour"] == 1000.0
    assert entry["cost_per_1k_tokens_usd"] == 10.0
