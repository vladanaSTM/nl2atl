#!/usr/bin/env python
"""
Run a single experiment.
"""
import argparse

from ..config import Config, ExperimentCondition
from ..experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key (e.g., qwen-3b)")
    parser.add_argument("--finetuned", action="store_true", help="Use fine-tuned model")
    parser.add_argument(
        "--few_shot", action="store_true", help="Use few-shot prompting"
    )
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--experiments_config", default="configs/experiments.yaml")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.models_config, args.experiments_config)

    # Create condition
    condition_name = (
        f"{'finetuned' if args.finetuned else 'baseline'}_"
        f"{'few_shot' if args.few_shot else 'zero_shot'}"
    )
    condition = ExperimentCondition(
        name=condition_name, finetuned=args.finetuned, few_shot=args.few_shot
    )

    # Run
    runner = ExperimentRunner(config)
    runner.run_single_experiment(args.model, condition)


if __name__ == "__main__":
    main()
