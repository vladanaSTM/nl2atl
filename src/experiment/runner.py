"""Core experiment orchestration."""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from ..config import Config, ModelConfig, ExperimentCondition
from ..data_utils import get_output_options
from ..evaluation.exact_match import ExactMatchEvaluator
from ..models.few_shot import format_prompt
from ..models.registry import load_model, get_model_type, clear_gpu_memory
from ..models.utils import resolve_model_key

from .data_manager import ExperimentDataManager
from .reporter import ExperimentReporter, sha256_file


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

    def _build_adapter_run_name(self, model_key: str) -> str:
        """Build the shared fine-tuned adapter run name for a model and seed."""
        model_config = self.config.models[model_key]
        return f"{model_config.short_name}_finetuned{self._seed_suffix()}"

    def _adapter_final_path(self, model_key: str) -> Path:
        """Get the shared adapter path for a model and seed."""
        return (
            Path(self.config.models_dir)
            / self._build_adapter_run_name(model_key)
            / "final"
        )

    def _set_global_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

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
    def _cuda_supports_tf32() -> bool:
        """Return whether the active CUDA device supports TF32 kernels."""
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8

    @staticmethod
    def _cuda_supports_bf16() -> bool:
        """Return whether the active CUDA device supports BF16 training."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    @staticmethod
    def _training_dataset(
        items: List[Dict[str, Any]],
        model_type: str,
        tokenizer: Any,
        max_seq_length: Optional[int] = None,
    ) -> Dataset:
        """Build prompt/completion pairs consumed by TRL's SFTTrainer."""
        prompts = []
        completions = []
        for item in items:
            outputs = get_output_options(item)
            if not outputs:
                raise ValueError(f"Training item is missing output: {item.get('id')}")
            for output in outputs:
                prompt = format_prompt(
                    item["input"],
                    output_text=None,
                    few_shot=False,
                    model_type=model_type,
                    tokenizer=tokenizer,
                )
                full_prompt = format_prompt(
                    item["input"],
                    output,
                    few_shot=False,
                    model_type=model_type,
                    tokenizer=tokenizer,
                )
                if not full_prompt.startswith(prompt):
                    raise ValueError(
                        "Training prompt template did not produce a prompt prefix; "
                        f"cannot isolate completion for item {item.get('id')}"
                    )
                if tokenizer is not None and max_seq_length is not None:
                    prompt_tokens = tokenizer(prompt, add_special_tokens=False)[
                        "input_ids"
                    ]
                    full_tokens = tokenizer(full_prompt, add_special_tokens=False)[
                        "input_ids"
                    ]
                    if len(prompt_tokens) >= max_seq_length:
                        raise ValueError(
                            "Training prompt exceeds max_seq_length before the "
                            f"completion starts for item {item.get('id')}: "
                            f"prompt_tokens={len(prompt_tokens)}, "
                            f"max_seq_length={max_seq_length}"
                        )
                    if len(full_tokens) > max_seq_length:
                        raise ValueError(
                            "Training prompt+completion exceeds max_seq_length for "
                            f"item {item.get('id')}: full_tokens={len(full_tokens)}, "
                            f"max_seq_length={max_seq_length}"
                        )
                prompts.append(prompt)
                completions.append(full_prompt[len(prompt) :])
        return Dataset.from_dict({"prompt": prompts, "completion": completions})

    def _training_args(
        self,
        model_config: ModelConfig,
        output_dir: Path,
        max_steps: int,
    ) -> SFTConfig:
        """Build Hugging Face training arguments from global and model config."""
        train_batch_size = model_config.train_batch_size or self.config.batch_size
        eval_batch_size = model_config.eval_batch_size or train_batch_size
        grad_accum_steps = (
            model_config.gradient_accumulation_steps
            or self.config.gradient_accumulation_steps
        )

        return SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_strategy="steps",
            logging_steps=10,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=self.config.bf16 and self._cuda_supports_bf16(),
            optim=self.config.optim,
            report_to="none",
            max_grad_norm=self.config.max_grad_norm,
            seed=self.config.seed,
            data_seed=self.config.seed,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=max_steps,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            group_by_length=self.config.group_by_length,
            tf32=self.config.tf32 and self._cuda_supports_tf32(),
            max_length=model_config.max_seq_length,
            completion_only_loss=True,
            packing=self.config.packing,
            dataset_text_field="text",
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
            self.train_data_aug, model_type, tokenizer, model_config.max_seq_length
        )
        val_dataset = self._training_dataset(
            self.val_data, model_type, tokenizer, model_config.max_seq_length
        )

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

    def _ensure_shared_adapter(
        self,
        model_key: str,
        model_config: ModelConfig,
        model_type: str,
        overwrite: bool,
    ) -> Tuple[str, bool, Optional[float]]:
        """Train or reuse the shared adapter for all fine-tuned conditions."""
        self._assert_finetune_allowed(model_config)

        adapter_path = self._adapter_final_path(model_key)
        if adapter_path.exists() and not overwrite:
            print(f"Reusing fine-tuned adapter at {adapter_path}")
            return str(adapter_path), True, None

        start_time = time.perf_counter()
        trained_path = self._train_model(
            model_config,
            model_type,
            self._build_adapter_run_name(model_key),
        )
        duration = round(time.perf_counter() - start_time, 2)
        return trained_path, False, duration

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

    def _record_result(
        self,
        model_config: ModelConfig,
        condition: ExperimentCondition,
        effective_finetuned: bool,
        run_name: str,
        metrics: Dict[str, Any],
        detailed_results: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Persist one evaluated condition and record its metadata."""
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
        split_manifest_path = self.reporter.save_split_manifest(
            run_name=run_name,
            config=self.config,
            dataset_path=str(self.config.data_path),
            train_data=self.data_manager.train_data or [],
            val_data=self.val_data,
            test_data=self.test_data,
        )
        metadata["split_manifest_path"] = str(split_manifest_path)
        metadata["split_manifest_sha256"] = sha256_file(split_manifest_path)
        metadata["metrics"] = metrics
        if extra_metadata:
            metadata.update(extra_metadata)

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
        return result

    def _run_condition(
        self,
        model_key: str,
        condition: ExperimentCondition,
        adapter_path: Optional[str] = None,
        adapter_metadata: Optional[Dict[str, Any]] = None,
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

        model = None
        tokenizer = None

        try:
            adapter_reused = adapter_path is not None
            if effective_finetuned and adapter_path is None:
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

            metadata = dict(adapter_metadata or {})
            if effective_finetuned:
                metadata.update(
                    {
                        "adapter_path": adapter_path,
                        "adapter_reused": adapter_reused,
                    }
                )

            result = self._record_result(
                model_config=model_config,
                condition=condition,
                effective_finetuned=effective_finetuned,
                run_name=run_name,
                metrics=metrics,
                detailed_results=detailed_results,
                extra_metadata=metadata,
            )

        finally:
            if not (self.reuse_models and not effective_finetuned):
                del model
                del tokenizer
                clear_gpu_memory()

            self.reporter.finalize()

        return result

    def run_model_experiments(
        self,
        model_key: str,
        conditions: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Run selected conditions for one model, sharing fine-tuning work."""
        model_key = resolve_model_key(model_key, self.config.models)
        model_config = self.config.models[model_key]
        model_type = get_model_type(model_config.name)

        if conditions is None:
            run_conditions = list(self.config.conditions)
        else:
            requested = set(conditions)
            run_conditions = [c for c in self.config.conditions if c.name in requested]
            missing = requested.difference({c.name for c in run_conditions})
            if missing:
                raise ValueError(f"No conditions matched: {sorted(missing)}")

        baseline_conditions = [c for c in run_conditions if not c.finetuned]
        finetuned_conditions = [c for c in run_conditions if c.finetuned]

        for condition in baseline_conditions:
            run_name = self._build_run_name(model_key, condition)
            result_path = self.reporter.get_result_path(run_name)
            if result_path.exists() and not overwrite:
                print(f"Skipping {run_name}: results exist at {result_path}")
                continue
            self._run_condition(model_key, condition)

        if not finetuned_conditions:
            self._release_model_cache()
            return

        self._assert_finetune_allowed(model_config)

        missing_finetuned = []
        for condition in finetuned_conditions:
            run_name = self._build_run_name(model_key, condition)
            result_path = self.reporter.get_result_path(run_name)
            if result_path.exists() and not overwrite:
                print(f"Skipping {run_name}: results exist at {result_path}")
                continue
            missing_finetuned.append(condition)

        if not missing_finetuned:
            self._release_model_cache()
            return

        adapter_path, adapter_reused, training_duration = self._ensure_shared_adapter(
            model_key=model_key,
            model_config=model_config,
            model_type=model_type,
            overwrite=overwrite,
        )
        adapter_metadata = {
            "adapter_path": adapter_path,
            "adapter_reused": adapter_reused,
            "shared_adapter_run_id": self._build_adapter_run_name(model_key),
        }
        if training_duration is not None:
            adapter_metadata["adapter_training_duration_seconds"] = training_duration

        model = None
        tokenizer = None
        try:
            model, tokenizer = self._load_or_reuse_model(
                model_config,
                adapter_path,
                effective_finetuned=True,
            )
            for condition in missing_finetuned:
                run_name = self._build_run_name(model_key, condition)

                print(f"\n{'='*60}")
                print(f"Running: {run_name}")
                print(f"{'='*60}")

                self.reporter.start_timer()
                metrics, detailed_results = self._evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    model_type=model_type,
                    condition=condition,
                )
                self._record_result(
                    model_config=model_config,
                    condition=condition,
                    effective_finetuned=True,
                    run_name=run_name,
                    metrics=metrics,
                    detailed_results=detailed_results,
                    extra_metadata=adapter_metadata,
                )
        finally:
            del model
            del tokenizer
            clear_gpu_memory()
            self.reporter.finalize()
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
