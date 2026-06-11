"""Experiment reporting and logging."""

import hashlib
import json
import subprocess
import sys
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


def sha256_file(path: Path) -> Optional[str]:
    """Return a SHA-256 digest for a file, or None when unavailable."""
    try:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _stable_item_id(item: Dict[str, Any], index: int) -> str:
    if item.get("id") is not None:
        return str(item["id"])
    payload = {
        "index": index,
        "input": item.get("input"),
        "outputs": (
            item.get("outputs") or item.get("expected_options") or item.get("expected")
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _split_entries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = []
    for index, item in enumerate(items):
        input_text = str(item.get("input", ""))
        outputs = item.get("outputs") or item.get("expected_options") or []
        encoded_outputs = json.dumps(outputs, sort_keys=True, ensure_ascii=False)
        entries.append(
            {
                "id": _stable_item_id(item, index),
                "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                "outputs_sha256": hashlib.sha256(
                    encoded_outputs.encode("utf-8")
                ).hexdigest(),
            }
        )
    return entries


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

    def __init__(
        self, output_dir: Path, predictions_subdir: Optional[str] = None
    ) -> None:
        self.output_dir = Path(output_dir)
        # When set (e.g. "smoke_test"), results and split manifests are nested
        # under this subfolder so throwaway smoke runs never mix with or get
        # aggregated alongside real prediction files.
        self.predictions_subdir = predictions_subdir
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
            "model_revision": model_config.revision,
            "condition": condition.name,
            "seed": config.seed,
            "split_seed": (
                config.split_seed if config.split_seed is not None else config.seed
            ),
            "stratify": config.stratify,
            "cv_folds": config.cv_folds,
            "cv_fold": config.cv_fold,
            "finetuned": effective_finetuned,
            "few_shot": condition.few_shot,
            "num_epochs": config.num_epochs if effective_finetuned else 0,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "bf16": config.bf16,
            "tf32": config.tf32,
            "lr_scheduler_type": config.lr_scheduler_type,
            "eval_strategy": config.eval_strategy,
            "save_strategy": config.save_strategy,
            "save_total_limit": config.save_total_limit,
            "batch_size": config.batch_size,
            "train_batch_size": model_config.train_batch_size or config.batch_size,
            "eval_batch_size": model_config.eval_batch_size
            or model_config.train_batch_size
            or config.batch_size,
            "gradient_accumulation_steps": model_config.gradient_accumulation_steps
            or config.gradient_accumulation_steps,
            "max_seq_length": model_config.max_seq_length,
            "load_in_4bit": model_config.load_in_4bit,
            "lora_r": model_config.lora_r,
            "lora_alpha": model_config.lora_alpha,
            "lora_dropout": model_config.lora_dropout,
            "target_modules": list(model_config.target_modules),
            "optim": config.optim,
            "gradient_checkpointing": config.gradient_checkpointing,
            "dataloader_num_workers": config.dataloader_num_workers,
            "dataloader_pin_memory": config.dataloader_pin_memory,
            "group_by_length": config.group_by_length,
            "max_grad_norm": config.max_grad_norm,
            "packing": config.packing,
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
            "dataset_sha256": sha256_file(Path(dataset_path)),
            "total_samples": total_samples,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "models_config_path": config.models_config_path,
            "models_config_sha256": (
                sha256_file(Path(config.models_config_path))
                if config.models_config_path
                else None
            ),
            "experiments_config_path": config.experiments_config_path,
            "experiments_config_sha256": (
                sha256_file(Path(config.experiments_config_path))
                if config.experiments_config_path
                else None
            ),
            "pyproject_sha256": sha256_file(Path("pyproject.toml")),
            "uv_lock_sha256": sha256_file(Path("uv.lock")),
            "command_argv": list(sys.argv),
            **self._build_run_config(
                config, model_config, condition, effective_finetuned
            ),
            **latency_stats,
        }

        # Add timing info if available
        if self._timer:
            metadata.update(self._timer.to_dict())

        return metadata

    def save_split_manifest(
        self,
        run_name: str,
        config: Config,
        dataset_path: str,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
    ) -> Path:
        """Save the exact split membership used by a run."""
        manifest_dir = self.output_dir / "split_manifests"
        if self.predictions_subdir:
            manifest_dir = manifest_dir / self.predictions_subdir
        manifest_path = manifest_dir / f"{run_name}.json"
        split_seed = config.split_seed if config.split_seed is not None else config.seed
        if config.cv_folds and config.cv_folds >= 2 and config.cv_fold is not None:
            split_algorithm = (
                f"stratified {config.cv_folds}-fold CV; held-out fold "
                f"{config.cv_fold} = test; remaining folds = train+val"
            )
        elif config.stratify:
            split_algorithm = (
                "stratified random.Random(split_seed) shuffle; round counts"
            )
        else:
            split_algorithm = "random.Random(split_seed).shuffle; round counts"
        manifest = {
            "created_at": get_utc_timestamp(),
            "run_id": run_name,
            "dataset_path": dataset_path,
            "dataset_sha256": sha256_file(Path(dataset_path)),
            "split_algorithm": split_algorithm,
            "split_sizes": {
                "train_size": config.train_size,
                "val_size": config.val_size,
                "test_size": config.test_size,
                "augment_factor": config.augment_factor,
            },
            "seed": config.seed,
            "split_seed": split_seed,
            "stratify": config.stratify,
            "cv_folds": config.cv_folds,
            "cv_fold": config.cv_fold,
            "counts": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data),
            },
            "train": _split_entries(train_data),
            "validation": _split_entries(val_data),
            "test": _split_entries(test_data),
        }
        save_json(manifest, manifest_path)
        return manifest_path

    def get_result_path(self, run_name: str) -> Path:
        """Get path for saving results."""
        predictions_dir = self.output_dir / "model_predictions"
        if self.predictions_subdir:
            predictions_dir = predictions_dir / self.predictions_subdir
        return predictions_dir / f"{run_name}.json"

    def save_result(self, run_name: str, result: Dict[str, Any]) -> Path:
        """Save result to JSON file."""
        result_path = self.get_result_path(run_name)
        save_json(result, result_path)
        return result_path

    def finalize(self) -> None:
        """Finalize reporting state."""
        self.stop_timer()
