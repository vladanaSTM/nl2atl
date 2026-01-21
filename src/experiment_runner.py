"""
Main experiment runner with wandb integration.
"""

import os
import json
import wandb
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from .config import Config, ModelConfig, ExperimentCondition
from .data_utils import load_data, split_data, augment_data
from .model_registry import load_model, get_model_type, clear_gpu_memory
from .exact_match_evaluator import ExactMatchEvaluator
from .few_shot import format_prompt


class ExperimentRunner:
    """Orchestrates all experiments with tracking."""

    def __init__(self, config: Config):
        self.config = config
        self.evaluator = ExactMatchEvaluator()
        self.all_results = []

        # Load and split data once
        self.data = load_data(config.data_path)
        self.train_data, self.val_data, self.test_data = split_data(
            self.data,
            test_size=config.test_size,
            val_size=config.val_size,
            seed=config.seed,
        )

        # Augment training data
        self.train_data_aug = augment_data(self.train_data, config.augment_factor)

        # Optional reuse of loaded models across conditions (set REUSE_MODEL_CACHE=0 to disable)
        self.model_cache = {}
        self.reuse_models = os.getenv("REUSE_MODEL_CACHE", "1") != "0"

        print(
            f"Data loaded: Train={len(self.train_data_aug)}, Val={len(self.val_data)}, Test={len(self.test_data)}"
        )

    def _resolve_model_key(self, model_arg: str) -> str:
        """Resolve user-provided model argument to a config key.

        Accepts config key, short_name, or full HF repo name (case-insensitive).
        """
        if model_arg in self.config.models:
            return model_arg

        def normalize(token: str) -> str:
            token = token.lower()
            for prefix in "azure-":
                if token.startswith(prefix):
                    token = token[len(prefix) :]
            return token

        needle = model_arg.lower()
        normalized_needle = normalize(model_arg)
        for key, mc in self.config.models.items():
            if (
                mc.short_name.lower() == needle
                or mc.name.lower() == needle
                or normalize(mc.short_name) == normalized_needle
                or normalize(mc.name) == normalized_needle
                or normalize(key) == normalized_needle
            ):
                return key

        raise KeyError(model_arg)

    def run_single_experiment(
        self, model_key: str, condition: ExperimentCondition, save_model: bool = True
    ) -> Dict:
        """Run a single experiment (one model, one condition)."""
        model_key = self._resolve_model_key(model_key)
        model_config = self.config.models[model_key]
        model_type = get_model_type(model_config.name)
        is_azure = model_config.provider.lower() == "azure"
        if condition.finetuned and is_azure:
            print(
                f"Skipping {model_config.short_name} ({condition.name}): fine-tuning disabled for provider=azure."
            )
            return {
                "run_name": f"{model_config.short_name}_{condition.name}",
                "model": model_config.short_name,
                "condition": condition.name,
                "finetuned": condition.finetuned,
                "few_shot": condition.few_shot,
                "skipped": True,
                "skip_reason": "finetuning_disabled_for_azure",
            }
        effective_finetuned = condition.finetuned

        run_name = f"{model_config.short_name}_{condition.name}"
        print(f"\n{'='*60}")
        print(f"Running: {run_name}")
        print(f"{'='*60}")

        # Initialize wandb run
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config={
                "model": model_config.name,
                "model_short": model_config.short_name,
                "condition": condition.name,
                "finetuned": effective_finetuned,
                "few_shot": condition.few_shot,
                "num_epochs": self.config.num_epochs if effective_finetuned else 0,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_few_shot": (
                    self.config.num_few_shot_examples if condition.few_shot else 0
                ),
            },
            reinit=True,
        )

        adapter_path = None

        # Train if needed
        if effective_finetuned:
            adapter_path = self._train_model(model_config, model_type, run_name)

        cache_key = (model_config.name, adapter_path or "base")
        reused = False
        model = tokenizer = None

        if (
            self.reuse_models
            and not effective_finetuned
            and cache_key in self.model_cache
        ):
            model, tokenizer = self.model_cache[cache_key]
            reused = True
            print("Reusing cached model from GPU memory for this condition")
        else:
            model, tokenizer = load_model(
                model_config, for_training=False, load_adapter=adapter_path
            )
            if self.reuse_models and not effective_finetuned:
                self.model_cache[cache_key] = (model, tokenizer)

        # Evaluate
        metrics = self.evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            test_data=self.test_data,
            model_type=model_type,
            few_shot=condition.few_shot,
            num_few_shot=self.config.num_few_shot_examples,
            verbose=True,
        )

        # Log metrics to wandb
        wandb.log(
            {"eval/exact_match": metrics["exact_match"]},
            commit=False,
        )

        # Log detailed predictions as a table (Table constructor with columns + data per wandb guidance)
        wandb.run.summary["num_predictions"] = len(self.evaluator.results)

        table_columns = [
            "Example_ID",
            "Input",
            "Expected_Output",
            "Generated_Output",
            "Exact_Match",
        ]

        table_rows = []
        if not self.evaluator.results:
            print("No predictions generated; logging empty predictions table to wandb.")
        else:
            for idx, result in enumerate(self.evaluator.results):
                table_rows.append(
                    [
                        idx + 1,
                        result["input"],
                        result["expected"],
                        result["generated"],
                        result["exact_match"],
                    ]
                )

        predictions_table = wandb.Table(columns=table_columns, data=table_rows)
        wandb.log({"predictions_table": predictions_table}, commit=True)

        predictions_artifact = wandb.Artifact(
            name=f"{run_name}-predictions", type="predictions"
        )
        predictions_artifact.add(predictions_table, "predictions")
        wandb.log_artifact(predictions_artifact)

        wandb.run.summary["predictions_table_rows"] = len(table_rows)
        wandb.run.summary["predictions_table_artifact"] = predictions_artifact.name

        # Print summary
        print(f"\nResults for {run_name}:")
        print(f"  Exact Match:    {metrics['exact_match']:.1%}")

        # Save results
        result = {
            "run_name": run_name,
            "model": model_config.short_name,
            "condition": condition.name,
            "finetuned": effective_finetuned,
            "few_shot": condition.few_shot,
            "metrics": metrics,
            "detailed_results": self.evaluator.results,
        }

        self.all_results.append(result)

        # Save to file
        result_path = (
            Path(self.config.output_dir) / "model_predictions" / f"{run_name}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)

        # Clean up if not caching this model
        if not (self.reuse_models and not effective_finetuned):
            del model, tokenizer
            clear_gpu_memory()

        wandb.finish()

        return result

    def _train_model(
        self, model_config: ModelConfig, model_type: str, run_name: str
    ) -> str:
        """Train a model and return the adapter path."""

        if model_config.provider.lower() == "azure":
            raise ValueError("Training is disabled for Azure models.")

        print(f"\nTraining {model_config.short_name}...")

        # Load model for training
        model, tokenizer = load_model(model_config, for_training=True)

        # Optional short-run probe to test throughput/memory without full epochs
        max_steps_env = os.getenv("TRAIN_MAX_STEPS")
        max_steps = -1  # HF/TRL expect -1 for full training, not None
        if max_steps_env:
            try:
                parsed = int(max_steps_env)
                max_steps = parsed if parsed > 0 else -1
            except ValueError:
                print(f"Warning: ignoring non-integer TRAIN_MAX_STEPS={max_steps_env}")

        # Prepare dataset
        train_dataset = Dataset.from_dict(
            {
                "text": [
                    format_prompt(
                        item["input"],
                        item["output"],
                        few_shot=False,
                        model_type=model_type,
                        tokenizer=tokenizer,
                    )
                    for item in self.train_data_aug
                ]
            }
        )

        val_dataset = Dataset.from_dict(
            {
                "text": [
                    format_prompt(
                        item["input"],
                        item["output"],
                        few_shot=False,
                        model_type=model_type,
                        tokenizer=tokenizer,
                    )
                    for item in self.val_data
                ]
            }
        )

        # Output directory
        output_dir = Path(self.config.models_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=self.config.bf16,
            optim="adamw_torch_fused",
            report_to="wandb",
            max_grad_norm=0.3,
            seed=self.config.seed,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=max_steps,
            dataloader_num_workers=4,
            group_by_length=True,
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        trainer.train()

        # Save
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # Clean up
        del model, tokenizer, trainer
        clear_gpu_memory()

        return str(final_path)

    def run_all_experiments(
        self,
        models: List[str] = None,
        conditions: List[str] = None,
        model_provider: str = "hf",
    ):
        """Run all experiments."""

        if models is None:
            models = list(self.config.models.keys())

        if model_provider not in {"hf", "azure", "all"}:
            raise ValueError(
                f"Invalid model_provider '{model_provider}'. Use 'hf', 'azure', or 'all'."
            )

        if model_provider != "all":
            filtered_models = []
            for model_key in models:
                resolved_key = self._resolve_model_key(model_key)
                provider = self.config.models[resolved_key].provider.lower()
                is_azure = provider == "azure"
                if (model_provider == "azure" and is_azure) or (
                    model_provider == "hf" and not is_azure
                ):
                    filtered_models.append(model_key)
            models = filtered_models

        if conditions is None:
            conditions = self.config.conditions
        else:
            conditions = [c for c in self.config.conditions if c.name in conditions]

        total = len(models) * len(conditions)
        current = 0

        print(f"\n{'='*60}")
        print(f"RUNNING {total} EXPERIMENTS")
        print(f"Models: {models}")
        print(f"Conditions: {[c.name for c in conditions]}")
        print(f"{'='*60}")

        for model_key in models:
            for condition in conditions:
                current += 1
                print(f"\n[{current}/{total}] ", end="")

                try:
                    resolved_key = self._resolve_model_key(model_key)
                    self.run_single_experiment(resolved_key, condition)
                except Exception as e:
                    print(f"ERROR: {e}")
                    wandb.finish(exit_code=1)
                    continue

        # Release any cached models
        self._release_model_cache()

    def _release_model_cache(self):
        """Free any cached models to release GPU/CPU memory."""
        if not self.model_cache:
            return
        for _, (model, tokenizer) in self.model_cache.items():
            del model, tokenizer
        self.model_cache.clear()
        clear_gpu_memory()
