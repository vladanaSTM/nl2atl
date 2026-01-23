"""Experiment reporting and logging."""

from pathlib import Path
from typing import Dict, Any, Optional, List

import wandb

from ..config import ModelConfig, ExperimentCondition, Config
from ..io_utils import save_json


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
        }

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
        wandb.finish()
