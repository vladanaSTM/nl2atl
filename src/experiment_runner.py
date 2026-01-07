"""
Main experiment runner with wandb integration.
"""
import os
import json
import wandb
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from .config import Config, ModelConfig, ExperimentCondition
from .data_utils import load_data, split_data, augment_data
from .model_registry import load_model, get_model_type, clear_gpu_memory
from .evaluator import ATLEvaluator
from .few_shot import format_prompt


class ExperimentRunner:
    """Orchestrates all experiments with tracking."""
    
    def __init__(self, config: Config):
        self.config = config
        self.evaluator = ATLEvaluator()
        self.all_results = []
        
        # Load and split data once
        self.data = load_data(config.data_path)
        self.train_data, self.val_data, self.test_data = split_data(
            self.data,
            test_size=config.test_size,
            val_size=config.val_size,
            seed=config.seed
        )
        
        # Augment training data
        self.train_data_aug = augment_data(self.train_data, config.augment_factor)
        
        print(f"Data loaded: Train={len(self.train_data_aug)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
    
    def run_single_experiment(
        self,
        model_key: str,
        condition: ExperimentCondition,
        save_model: bool = True
    ) -> Dict:
        """Run a single experiment (one model, one condition)."""
        
        model_config = self.config.models[model_key]
        model_type = get_model_type(model_config.name)
        
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
                "finetuned": condition.finetuned,
                "few_shot": condition.few_shot,
                "num_epochs": self.config.num_epochs if condition.finetuned else 0,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_few_shot": self.config.num_few_shot_examples if condition.few_shot else 0,
            },
            reinit=True
        )
        
        adapter_path = None
        
        # Train if needed
        if condition.finetuned:
            adapter_path = self._train_model(model_config, model_type, run_name)
        
        # Load model for evaluation
        model, tokenizer = load_model(
            model_config,
            for_training=False,
            load_adapter=adapter_path
        )
        
        # Evaluate
        metrics = self.evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            test_data=self.test_data,
            model_type=model_type,
            few_shot=condition.few_shot,
            num_few_shot=self.config.num_few_shot_examples,
            verbose=True
        )
        
        # Log metrics to wandb
        wandb.log({
            "eval/exact_match": metrics['exact_match'],
            "eval/agent_f1": metrics['agent_f1'],
            "eval/temporal_f1": metrics['temporal_f1'],
            "eval/logical_f1": metrics['logical_f1'],
            "eval/syntax_valid": metrics['syntax_valid'],
            "eval/overall_score": metrics['overall_score'],
        })
        
        # Log detailed predictions as a table
        predictions_table = wandb.Table(columns=[
            "Example_ID",
            "Input",
            "Expected_Output",
            "Generated_Output",
            "Exact_Match",
            "Agent_F1",
            "Temporal_F1",
            "Logical_F1",
            "Syntax_Valid",
            "Overall_Score"
        ])
        
        for idx, result in enumerate(self.evaluator.results):
            predictions_table.add_data(
                idx + 1,
                result['input'],
                result['expected'],
                result['generated'],
                result['scores']['exact_match'],
                result['scores']['agent_f1'],
                result['scores']['temporal_f1'],
                result['scores']['logical_f1'],
                result['scores']['syntax_valid'],
                result['scores']['overall_score']
            )
        
        wandb.log({"predictions_table": predictions_table})
        
        # Print summary
        print(f"\nResults for {run_name}:")
        print(f"  Exact Match:    {metrics['exact_match']:.1%}")
        print(f"  Agent F1:       {metrics['agent_f1']:.1%}")
        print(f"  Temporal F1:    {metrics['temporal_f1']:.1%}")
        print(f"  Logical F1:     {metrics['logical_f1']:.1%}")
        print(f"  Syntax Valid:   {metrics['syntax_valid']:.1%}")
        print(f"  Overall Score:  {metrics['overall_score']:.1%}")
        
        # Save results
        result = {
            "run_name": run_name,
            "model": model_config.short_name,
            "condition": condition.name,
            "finetuned": condition.finetuned,
            "few_shot": condition.few_shot,
            "metrics": metrics,
            "detailed_results": self.evaluator.results
        }
        
        self.all_results.append(result)
        
        # Save to file
        result_path = Path(self.config.output_dir) / "results" / f"{run_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Clean up
        del model, tokenizer
        clear_gpu_memory()
        
        wandb.finish()
        
        return result
    
    def _train_model(self, model_config: ModelConfig, model_type: str, run_name: str) -> str:
        """Train a model and return the adapter path."""
        
        print(f"\nTraining {model_config.short_name}...")
        
        # Load model for training
        model, tokenizer = load_model(model_config, for_training=True)
        
        # Prepare dataset
        train_dataset = Dataset.from_dict({
            'text': [
                format_prompt(item['input'], item['output'], few_shot=False, model_type=model_type)
                for item in self.train_data_aug
            ]
        })
        
        val_dataset = Dataset.from_dict({
            'text': [
                format_prompt(item['input'], item['output'], few_shot=False, model_type=model_type)
                for item in self.val_data
            ]
        })
        
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
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=self.config.bf16,
            optim="paged_adamw_8bit",
            report_to="wandb",
            max_grad_norm=0.3,
            seed=self.config.seed,
            ddp_find_unused_parameters=False,
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
    
    def run_all_experiments(self, models: List[str] = None, conditions: List[str] = None):
        """Run all experiments."""
        
        if models is None:
            models = list(self.config.models.keys())
        
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
                    self.run_single_experiment(model_key, condition)
                except Exception as e:
                    print(f"ERROR: {e}")
                    wandb.finish(exit_code=1)
                    continue
        
        # Save all results
        self._save_summary()
    
    def _save_summary(self):
        """Save summary of all experiments."""
        summary_path = Path(self.config.output_dir) / "results" / "summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.all_results),
            "results": []
        }
        
        for result in self.all_results:
            summary["results"].append({
                "run_name": result["run_name"],
                "model": result["model"],
                "finetuned": result["finetuned"],
                "few_shot": result["few_shot"],
                "exact_match": result["metrics"]["exact_match"],
                "overall_score": result["metrics"]["overall_score"],
            })
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")