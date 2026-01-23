#!/usr/bin/env python
"""
Run all experiments for ATL formula generation comparison.
"""
import argparse
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from ..config import Config
from ..experiment import ExperimentRunner


def aggregate_seed_results(results):
    grouped = defaultdict(list)
    for result in results:
        metrics = result.get("metrics")
        if not metrics:
            continue
        key = (result["model"], result["condition"])
        grouped[key].append(
            {
                "seed": result.get("seed"),
                "exact_match": metrics.get("exact_match"),
                "n_examples": metrics.get("n_examples"),
            }
        )

    aggregates = []
    for (model, condition), items in grouped.items():
        values = [i["exact_match"] for i in items if i["exact_match"] is not None]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregates.append(
            {
                "model": model,
                "condition": condition,
                "num_seeds": len(values),
                "exact_match_mean": mean,
                "exact_match_std": std,
                "per_seed": items,
            }
        )

    return aggregates


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
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        help="Run all experiments even if output files already exist.",
    )
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--experiments_config", default="configs/experiments.yaml")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.models_config, args.experiments_config)

    seeds = config.resolve_seeds()
    all_results = []

    if len(seeds) == 1:
        runner = ExperimentRunner(config)
        runner.run_all_experiments(
            models=args.models,
            conditions=args.conditions,
            model_provider=args.model_provider,
            overwrite=args.overwrite,
        )
        all_results.extend(runner.all_results)
    else:
        print(f"Running {len(seeds)} seeds: {seeds}")
        for seed in seeds:
            seed_config = deepcopy(config)
            seed_config.seed = seed
            seed_config.seeds = seeds

            runner = ExperimentRunner(seed_config)
            runner.run_all_experiments(
                models=args.models,
                conditions=args.conditions,
                model_provider=args.model_provider,
                overwrite=args.overwrite,
            )
            all_results.extend(runner.all_results)

        aggregates = aggregate_seed_results(all_results)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "seed_aggregate_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2, ensure_ascii=False)

        print(f"Saved seed-aggregated metrics to {summary_path}")


if __name__ == "__main__":
    main()
