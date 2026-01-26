"""Experiment reporting and logging."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from decimal import Decimal, ROUND_HALF_UP
import time

import wandb

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

    def __init__(
        self,
        output_dir: Path,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
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

    def _build_wandb_config(
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
            "price_input_per_1k": model_config.price_input_per_1k,
            "price_output_per_1k": model_config.price_output_per_1k,
            "price_input_per_token": (
                model_config.price_input_per_1k / 1000.0
                if model_config.price_input_per_1k is not None
                else None
            ),
            "price_output_per_token": (
                model_config.price_output_per_1k / 1000.0
                if model_config.price_output_per_1k is not None
                else None
            ),
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
            latency_stats = {
                "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
                "latency_min_ms": round(min(latencies), 2),
                "latency_max_ms": round(max(latencies), 2),
                "latency_total_ms": round(sum(latencies), 2),
            }

        cost_stats = {}
        total_cost = sum(Decimal(str(r.get("cost_usd", 0))) for r in results)
        if total_cost > 0:
            q = Decimal("0.000001")
            n_dec = Decimal(len(results)) if results else Decimal("1")
            cost_stats = {
                "cost_total_usd": float(total_cost.quantize(q, rounding=ROUND_HALF_UP)),
                "cost_input_usd": float(
                    sum(
                        Decimal(str(r.get("cost_input_usd", 0))) for r in results
                    ).quantize(q, rounding=ROUND_HALF_UP)
                ),
                "cost_output_usd": float(
                    sum(
                        Decimal(str(r.get("cost_output_usd", 0))) for r in results
                    ).quantize(q, rounding=ROUND_HALF_UP)
                ),
                "avg_cost_usd": float(
                    (total_cost / n_dec).quantize(q, rounding=ROUND_HALF_UP)
                ),
                "avg_cost_input_usd": float(
                    (
                        sum(Decimal(str(r.get("cost_input_usd", 0))) for r in results)
                        / n_dec
                    ).quantize(q, rounding=ROUND_HALF_UP)
                ),
                "avg_cost_output_usd": float(
                    (
                        sum(Decimal(str(r.get("cost_output_usd", 0))) for r in results)
                        / n_dec
                    ).quantize(q, rounding=ROUND_HALF_UP)
                ),
            }

        metadata = {
            "run_id": run_name,
            "git_commit": self._git_commit,
            "dataset_path": dataset_path,
            "total_samples": total_samples,
            "successful_predictions": successful,
            "failed_predictions": failed,
            **self._build_wandb_config(
                config, model_config, condition, effective_finetuned
            ),
            **latency_stats,
            **cost_stats,
        }

        # Add timing info if available
        if self._timer:
            metadata.update(self._timer.to_dict())

        return metadata

    def init_wandb_run(
        self,
        config: Config,
        run_name: str,
        model_config: ModelConfig,
        condition: ExperimentCondition,
        effective_finetuned: bool,
    ) -> None:
        """Initialize a W&B run."""
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name,
            config=self._build_wandb_config(
                config, model_config, condition, effective_finetuned
            ),
            reinit=True,
        )

    def log_predictions_table(
        self, results: List[Dict[str, Any]], run_name: str
    ) -> None:
        """Log predictions to W&B as a table."""
        wandb.run.summary["num_predictions"] = len(results)

        columns = [
            "Example_ID",
            "Input",
            "Expected_Output",
            "Generated_Output",
            "Difficulty",
            "Exact_Match",
            "Latency_ms",
        ]

        rows = []
        for idx, result in enumerate(results):
            rows.append(
                [
                    idx + 1,
                    result["input"],
                    result["expected"],
                    result["generated"],
                    result.get("difficulty"),
                    result["exact_match"],
                    result.get("latency_ms"),
                ]
            )

        if not rows:
            print("No predictions generated; logging empty table to W&B.")

        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"predictions_table": table}, commit=True)

        artifact = wandb.Artifact(name=f"{run_name}-predictions", type="predictions")
        artifact.add(table, "predictions")
        wandb.log_artifact(artifact)

        wandb.run.summary["predictions_table_rows"] = len(rows)
        wandb.run.summary["predictions_table_artifact"] = artifact.name

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to W&B."""
        if "exact_match" in metrics:
            wandb.log({"eval/exact_match": metrics["exact_match"]}, commit=False)

        # Log latency metrics if present
        for key in ["latency_mean_ms", "latency_total_ms"]:
            if key in metrics:
                wandb.log({f"perf/{key}": metrics[key]}, commit=False)

        for key in [
            "total_cost_usd",
            "total_cost_input_usd",
            "total_cost_output_usd",
            "avg_cost_usd",
            "avg_cost_input_usd",
            "avg_cost_output_usd",
        ]:
            if key in metrics:
                wandb.log({f"cost/{key}": metrics[key]}, commit=False)

    def get_result_path(self, run_name: str) -> Path:
        """Get path for saving results."""
        return self.output_dir / "model_predictions" / f"{run_name}.json"

    def save_result(self, run_name: str, result: Dict[str, Any]) -> Path:
        """Save result to JSON file."""
        result_path = self.get_result_path(run_name)
        save_json(result, result_path)
        return result_path

    def finalize(self) -> None:
        """Finalize reporting (close W&B run, etc.)."""
        self.stop_timer()
        wandb.finish()
