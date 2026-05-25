"""Core experiment orchestration."""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from ..config import Config, ModelConfig, ExperimentCondition
from ..constants import Provider
from ..data_utils import get_output_options
from ..evaluation.exact_match import ExactMatchEvaluator
from ..models.few_shot import format_prompt
from ..models.registry import load_model, get_model_type, clear_gpu_memory
from ..models.utils import resolve_model_key

from .data_manager import ExperimentDataManager
from .reporter import ExperimentReporter


class ExperimentRunner:
    """Orchestrates all experiments with tracking."""

    def __init__(
        self,
        config: Config,
        data_manager: Optional[ExperimentDataManager] = None,
        reporter: Optional[ExperimentReporter] = None,
    ):
        self.config = config
        self.evaluator = ExactMatchEvaluator()
        self.all_results: List[Dict] = []

        self.data_manager = data_manager or ExperimentDataManager(
            data_path=Path(config.data_path),
            train_size=config.train_size,
            test_size=config.test_size,
            val_size=config.val_size,
            seed=config.seed,
            augment_factor=config.augment_factor,
        )
        self.reporter = reporter or ExperimentReporter(
            output_dir=Path(config.output_dir)
        )

        # Global seeding for reproducibility
        self._set_global_seed(self.config.seed)

        # Load and split data once
        (
            self.train_data_aug,
            self.val_data,
            self.test_data,
            self.data,
        ) = self.data_manager.prepare_data()

        # Model caching
        self.model_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
        self.reuse_models = os.getenv("REUSE_MODEL_CACHE", "1") != "0"

        train_count = len(self.data_manager.train_data or [])
        print(
            f"Data loaded: Train={train_count} "
            f"(augmented={len(self.train_data_aug)}), "
            f"Val={len(self.val_data)}, Test={len(self.test_data)}"
        )

    def _seed_suffix(self) -> str:
        """Get seed suffix for run naming."""
        if self.config.seeds and len(self.config.seeds) > 1:
            return f"_seed{self.config.seed}"
        return ""

    def _build_run_name(self, model_key: str, condition: ExperimentCondition) -> str:
        """Build a unique run name."""
        model_config = self.config.models[model_key]
        return f"{model_config.short_name}_{condition.name}{self._seed_suffix()}"

    def _set_global_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _assert_finetune_allowed(self, model_config: ModelConfig) -> None:
        """Check if finetuning is allowed for this model."""
        if model_config.is_azure:
            raise ValueError(
                f"Finetuning disabled for Azure models (model={model_config.short_name})."
            )
        if model_config.params_b is not None and model_config.params_b > 8:
            raise ValueError(
                f"Finetuning disabled for models >8B params "
                f"(model={model_config.short_name}, params_b={model_config.params_b})."
            )

    def _load_or_reuse_model(
        self,
        model_config: ModelConfig,
        adapter_path: Optional[str],
        effective_finetuned: bool,
    ) -> Tuple[Any, Any]:
        """Load model, reusing from cache if possible."""
        cache_key = (model_config.name, adapter_path or "base")

        if (
            self.reuse_models
            and not effective_finetuned
            and cache_key in self.model_cache
        ):
            model, tokenizer = self.model_cache[cache_key]
            print("Reusing cached model from GPU memory")
            return model, tokenizer

        model, tokenizer = load_model(
            model_config,
            for_training=False,
            load_adapter=adapter_path,
        )

        if self.reuse_models and not effective_finetuned:
            self.model_cache[cache_key] = (model, tokenizer)

        return model, tokenizer

    @staticmethod
    def _training_max_steps_from_env() -> int:
        """Return an optional short-run training limit from TRAIN_MAX_STEPS."""
        raw_value = os.getenv("TRAIN_MAX_STEPS")
        if not raw_value:
            return -1

        try:
            parsed = int(raw_value)
        except ValueError:
            print(f"Warning: ignoring non-integer TRAIN_MAX_STEPS={raw_value}")
            return -1

        return parsed if parsed > 0 else -1

    @staticmethod
    def _training_dataset(
        items: List[Dict[str, Any]],
        model_type: str,
        tokenizer: Any,
    ) -> Dataset:
        """Build the text dataset consumed by TRL's SFTTrainer."""
        prompts = []
        for item in items:
            outputs = get_output_options(item)
            if not outputs:
                raise ValueError(f"Training item is missing output: {item.get('id')}")
            for output in outputs:
                prompts.append(
                    format_prompt(
                        item["input"],
                        output,
                        few_shot=False,
                        model_type=model_type,
                        tokenizer=tokenizer,
                    )
                )
        return Dataset.from_dict({"text": prompts})

    def _training_args(
        self,
        model_config: ModelConfig,
        output_dir: Path,
        max_steps: int,
    ) -> TrainingArguments:
        """Build Hugging Face training arguments from global and model config."""
        train_batch_size = model_config.train_batch_size or self.config.batch_size
        eval_batch_size = model_config.eval_batch_size or train_batch_size
        grad_accum_steps = (
            model_config.gradient_accumulation_steps
            or self.config.gradient_accumulation_steps
        )

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
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
            report_to="none",
            max_grad_norm=0.3,
            seed=self.config.seed,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=max_steps,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            group_by_length=True,
            tf32=True,
        )

    def _train_model(
        self,
        model_config: ModelConfig,
        model_type: str,
        run_name: str,
    ) -> str:
        """Train a model and return the adapter path."""
        self._assert_finetune_allowed(model_config)
        print(f"\nTraining {model_config.short_name}...")

        model, tokenizer = load_model(model_config, for_training=True)
        train_dataset = self._training_dataset(
            self.train_data_aug, model_type, tokenizer
        )
        val_dataset = self._training_dataset(self.val_data, model_type, tokenizer)

        output_dir = Path(self.config.models_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        training_args = self._training_args(
            model_config=model_config,
            output_dir=output_dir,
            max_steps=self._training_max_steps_from_env(),
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        self._cleanup_training(trainer, model, tokenizer, train_dataset, val_dataset)

        return str(final_path)

    def _evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        model_config: ModelConfig,
        model_type: str,
        condition: ExperimentCondition,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluate the loaded model and return metrics plus detailed rows."""
        metrics = self.evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            test_data=self.test_data,
            model_type=model_type,
            few_shot=condition.few_shot,
            num_few_shot=self.config.num_few_shot_examples,
            verbose=True,
            price_input_per_1k=model_config.price_input_per_1k,
            price_output_per_1k=model_config.price_output_per_1k,
        )
        return metrics, list(self.evaluator.results)

    def _cleanup_training(
        self,
        trainer: SFTTrainer,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:
        """Clean up after training."""
        # Clear trainer references
        for attr in ("model", "model_wrapped", "optimizer", "lr_scheduler"):
            if hasattr(trainer, attr):
                setattr(trainer, attr, None)

        if hasattr(trainer, "accelerator"):
            try:
                trainer.accelerator.free_memory()
            except Exception:
                pass

        del trainer, train_dataset, val_dataset

        try:
            model.cpu()
        except Exception:
            pass

        del model, tokenizer
        clear_gpu_memory()

    def run_single_experiment(
        self,
        model_key: str,
        condition: ExperimentCondition,
        save_model: bool = True,
    ) -> Dict:
        """Run a single experiment (one model, one condition)."""
        model_key = resolve_model_key(model_key, self.config.models)
        model_config = self.config.models[model_key]
        model_type = get_model_type(model_config.name)

        if condition.finetuned:
            self._assert_finetune_allowed(model_config)

        effective_finetuned = condition.finetuned
        run_name = self._build_run_name(model_key, condition)

        print(f"\n{'='*60}")
        print(f"Running: {run_name}")
        print(f"{'='*60}")

        self.reporter.start_timer()

        adapter_path = None
        model = None
        tokenizer = None

        try:
            if effective_finetuned:
                adapter_path = self._train_model(model_config, model_type, run_name)

            model, tokenizer = self._load_or_reuse_model(
                model_config, adapter_path, effective_finetuned
            )
            metrics, detailed_results = self._evaluate_model(
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                model_type=model_type,
                condition=condition,
            )

            self.reporter.stop_timer()
            metadata = self.reporter.build_run_metadata(
                config=self.config,
                run_name=run_name,
                model_config=model_config,
                condition=condition,
                effective_finetuned=effective_finetuned,
                dataset_path=str(self.config.data_path),
                total_samples=len(self.test_data),
                results=detailed_results,
            )
            metadata["metrics"] = metrics

            print(f"\nResults for {run_name}:")
            print(f"  Exact Match: {metrics['exact_match']:.1%}")
            if "latency_mean_ms" in metadata:
                print(f"  Mean Latency: {metadata['latency_mean_ms']:.1f}ms")
            if metadata.get("duration_seconds"):
                print(f"  Total Duration: {metadata['duration_seconds']:.1f}s")

            result = {
                "metadata": metadata,
                "predictions": detailed_results,
            }

            self.all_results.append(result)
            self.reporter.save_result(run_name, result)

        finally:
            if not (self.reuse_models and not effective_finetuned):
                del model
                del tokenizer
                clear_gpu_memory()

            self.reporter.finalize()

        return result

    def run_all_experiments(
        self,
        models: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        model_provider: str = "hf",
        overwrite: bool = False,
    ) -> None:
        """Run all specified experiments."""
        if models is None:
            models = list(self.config.models.keys())

        if model_provider not in ("hf", "azure", "all"):
            raise ValueError(
                f"Invalid model_provider '{model_provider}'. Use 'hf', 'azure', or 'all'."
            )

        # Filter by provider
        if model_provider != "all":
            filtered = []
            for model_key in models:
                resolved_key = resolve_model_key(model_key, self.config.models)
                model_cfg = self.config.models[resolved_key]
                is_azure = model_cfg.is_azure

                if (model_provider == "azure" and is_azure) or (
                    model_provider == "hf" and not is_azure
                ):
                    filtered.append(model_key)
            models = filtered

        # Resolve conditions
        if conditions is None:
            run_conditions = self.config.conditions
        else:
            run_conditions = [c for c in self.config.conditions if c.name in conditions]

        total = len(models) * len(run_conditions)
        current = 0

        print(f"\n{'='*60}")
        print(f"RUNNING {total} EXPERIMENTS")
        print(f"Models: {models}")
        print(f"Conditions: {[c.name for c in run_conditions]}")
        print(f"{'='*60}")

        for model_key in models:
            for condition in run_conditions:
                current += 1
                print(f"\n[{current}/{total}] ", end="")

                try:
                    resolved_key = resolve_model_key(model_key, self.config.models)
                    run_name = self._build_run_name(resolved_key, condition)
                    result_path = self.reporter.get_result_path(run_name)

                    if result_path.exists() and not overwrite:
                        print(f"Skipping {run_name}: results exist at {result_path}")
                        continue

                    print(f"Running {run_name}: writing results to {result_path}")
                    self.run_single_experiment(resolved_key, condition)

                except Exception as e:
                    print(f"ERROR: {e}")
                    self.reporter.finalize()
                    clear_gpu_memory()

        self._release_model_cache()

    def _release_model_cache(self) -> None:
        """Free all cached models."""
        if not self.model_cache:
            return

        for _, (model, tokenizer) in self.model_cache.items():
            try:
                model.cpu()
            except Exception:
                pass
            del model, tokenizer

        self.model_cache.clear()
        clear_gpu_memory()

    def run(self) -> Dict[str, Any]:
        """Run all experiments with defaults and return results."""
        self.run_all_experiments()
        return {"results": self.all_results}


def run_experiment(config: Config) -> Dict[str, Any]:
    """Run experiment with given config."""
    runner = ExperimentRunner(config)
    return runner.run()
