"""Experiment reporting and logging."""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ModelConfig, ExperimentCondition, Config
from ..infra.io import save_json


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    """Compute percentile with linear interpolation."""
    if not values:
        return None
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    position = (percentile / 100.0) * (n - 1)
    lower = int(position)
    upper = min(lower + 1, n - 1)
    if lower == upper:
        return float(sorted_vals[lower])
    weight = position - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


class ExperimentTimer:
    """Context manager for timing experiments."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_timestamp: Optional[str] = None
        self.end_timestamp: Optional[str] = None

    def __enter__(self) -> "ExperimentTimer":
        self.start_time = time.perf_counter()
        self.start_timestamp = get_utc_timestamp()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.end_timestamp = get_utc_timestamp()

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_start": self.start_timestamp,
            "timestamp_end": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
        }


class ExperimentReporter:
    """Handles experiment logging, metrics, and result persistence."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._timer: Optional[ExperimentTimer] = None
        self._git_commit = get_git_commit()

    def start_timer(self) -> ExperimentTimer:
        """Start timing the experiment."""
        self._timer = ExperimentTimer()
        self._timer.__enter__()
        return self._timer

    def stop_timer(self) -> None:
        """Stop the experiment timer."""
        if self._timer:
            self._timer.__exit__(None, None, None)

    def _build_run_config(
        self,
        config: Config,
        model_config: ModelConfig,
        condition: ExperimentCondition,
        effective_finetuned: bool,
    ) -> Dict[str, Any]:
        return {
            "model": model_config.name,
            "model_short": model_config.short_name,
            "condition": condition.name,
            "seed": config.seed,
            "finetuned": effective_finetuned,
            "few_shot": condition.few_shot,
            "num_epochs": config.num_epochs if effective_finetuned else 0,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_few_shot": config.num_few_shot_examples if condition.few_shot else 0,
            "git_commit": self._git_commit,
        }

    def build_run_metadata(
        self,
        config: Config,
        run_name: str,
        model_config: ModelConfig,
        condition: ExperimentCondition,
        effective_finetuned: bool,
        dataset_path: str,
        total_samples: int,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build complete metadata for a run."""
        successful = sum(1 for r in results if r.get("generated"))
        failed = total_samples - successful

        # Compute latency statistics if available
        latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
        latency_stats = {}
        if latencies:
            p50 = _percentile(latencies, 50)
            p95 = _percentile(latencies, 95)
            p99 = _percentile(latencies, 99)
            latency_stats = {
                "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
                "latency_min_ms": round(min(latencies), 2),
                "latency_max_ms": round(max(latencies), 2),
                "latency_total_ms": round(sum(latencies), 2),
                "latency_p50_ms": round(p50, 2) if p50 is not None else None,
                "latency_p95_ms": round(p95, 2) if p95 is not None else None,
                "latency_p99_ms": round(p99, 2) if p99 is not None else None,
            }

        metadata = {
            "run_id": run_name,
            "git_commit": self._git_commit,
            "dataset_path": dataset_path,
            "total_samples": total_samples,
            "successful_predictions": successful,
            "failed_predictions": failed,
            **self._build_run_config(
                config, model_config, condition, effective_finetuned
            ),
            **latency_stats,
        }

        # Add timing info if available
        if self._timer:
            metadata.update(self._timer.to_dict())

        return metadata

    def get_result_path(self, run_name: str) -> Path:
        """Get path for saving results."""
        return self.output_dir / "model_predictions" / f"{run_name}.json"

    def save_result(self, run_name: str, result: Dict[str, Any]) -> Path:
        """Save result to JSON file."""
        result_path = self.get_result_path(run_name)
        save_json(result, result_path)
        return result_path

    def finalize(self) -> None:
        """Finalize reporting state."""
        self.stop_timer()
