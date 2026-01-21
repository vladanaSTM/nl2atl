#!/usr/bin/env python
"""
Run all experiments for ATL formula generation comparison.
"""
import argparse
from src.config import Config
from src.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None, help="Models to test (default: all)"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions to test (default: all)",
    )
    parser.add_argument(
        "--model_provider",
        choices=["hf", "azure", "all"],
        default="hf",
        help="Which model provider to run (default: hf).",
    )
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--experiments_config", default="configs/experiments.yaml")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.models_config, args.experiments_config)

    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all_experiments(
        models=args.models,
        conditions=args.conditions,
        model_provider=args.model_provider,
    )


if __name__ == "__main__":
    main()
