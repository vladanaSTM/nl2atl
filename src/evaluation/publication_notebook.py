"""Unified publication-focused evaluation notebook generation."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from ..infra.io import save_json

DEFAULT_PUBLICATION_NOTEBOOK_NAME = "publication_analysis.ipynb"


def _source_lines(text: str) -> list[str]:
    return [f"{line}\n" for line in dedent(text).strip("\n").splitlines()]


def _markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": _source_lines(text),
    }


def _code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "execution_count": None,
        "outputs": [],
        "source": _source_lines(text),
    }


def build_publication_notebook(eval_dir: Path, output_path: Path) -> None:
    """Write one compact notebook for AAAI-oriented result analysis."""
    eval_dir = Path(eval_dir)
    configured_eval_dir = json.dumps(eval_dir.resolve().as_posix())

    load_cell = """
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    CONFIGURED_EVAL_DIR = Path(__CONFIGURED_EVAL_DIR__)


    def resolve_eval_dir():
        candidates = [
            CONFIGURED_EVAL_DIR,
            Path.cwd(),
            Path.cwd() / "outputs" / "LLM-evaluation",
        ]
        for parent in Path.cwd().parents:
            candidates.append(parent / "outputs" / "LLM-evaluation")
        for candidate in candidates:
            if (candidate / "seed_aggregate_metrics_from_judged.json").exists():
                return candidate
            if (candidate / "efficiency_report.json").exists():
                return candidate
        return CONFIGURED_EVAL_DIR


    EVAL_DIR = resolve_eval_dir()


    def read_json(name, default):
        path = EVAL_DIR / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))


    aggregates = read_json("seed_aggregate_metrics_from_judged.json", [])
    efficiency = read_json("efficiency_report.json", {})
    agreement = read_json("agreement_report.json", {})
    manifest = read_json("reproducibility_manifest.json", {})

    summaries = []
    for summary_path in sorted(EVAL_DIR.glob("summary__judge-*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["_path"] = summary_path.name
        summaries.append(summary)

    print(f"Evaluation directory: {EVAL_DIR}")
    print(
        f"Loaded {len(aggregates)} aggregate groups, "
        f"{len(efficiency.get('models', []))} accuracy-latency rows, "
        f"{len(summaries)} judge summaries."
    )
    """.replace("__CONFIGURED_EVAL_DIR__", configured_eval_dir)

    transform_cell = """
    def metric_mean(metrics, name):
        value = (metrics or {}).get(name)
        if isinstance(value, dict):
            return value.get("mean")
        return value


    def metric_std(metrics, name):
        value = (metrics or {}).get(name)
        if isinstance(value, dict):
            return value.get("std")
        return None


    def metric_ci95(metrics, name):
        value = (metrics or {}).get(name)
        if isinstance(value, dict):
            return value.get("ci95")
        return None


    def judge_label(item):
        if item.get("judge_model"):
            return item.get("judge_model")
        judge_models = item.get("judge_models") or []
        return ", ".join(str(value) for value in judge_models) or "unknown"


    def aggregate_agreement(item, name):
        value = (item.get("judge_agreement") or {}).get(name)
        if isinstance(value, dict):
            return value.get("mean")
        return value


    aggregate_rows = []
    for item in aggregates:
        metrics = item.get("metrics") or {}
        aggregate_rows.append(
            {
                "model_short": item.get("model_short"),
                "condition": item.get("condition"),
                "judge_model": judge_label(item),
                "finetuned": item.get("finetuned"),
                "few_shot": item.get("few_shot"),
                "num_seeds": item.get("num_seeds"),
                "num_runs": item.get("num_runs"),
                "accuracy": metric_mean(metrics, "accuracy"),
                "accuracy_std": metric_std(metrics, "accuracy"),
                "accuracy_ci95": metric_ci95(metrics, "accuracy"),
                "exact_match_rate": metric_mean(metrics, "exact_match_rate"),
                "llm_approval_rate": metric_mean(metrics, "llm_approval_rate"),
                "accuracy_from_exact_match": metric_mean(metrics, "accuracy_from_exact_match"),
                "accuracy_boost_from_llm": metric_mean(metrics, "accuracy_boost_from_llm"),
                "latency_mean_ms": metric_mean(metrics, "latency_mean_ms"),
                "latency_p95_ms": metric_mean(metrics, "latency_p95_ms"),
                "judge_agreement_kappa": aggregate_agreement(item, "confidence_score"),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    if not aggregate_df.empty:
        aggregate_df = aggregate_df.sort_values(
            ["accuracy", "latency_mean_ms"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    efficiency_df = pd.DataFrame(efficiency.get("models", []))
    pareto_df = pd.DataFrame(efficiency.get("pareto_frontier", []))
    if not efficiency_df.empty:
        if "judge_model" not in efficiency_df.columns:
            efficiency_df["judge_model"] = None
        frontier_keys = set()
        if not pareto_df.empty:
            for _, row in pareto_df.iterrows():
                frontier_keys.add(
                    (
                        row.get("model") or row.get("model_short"),
                        row.get("condition"),
                        row.get("judge_model"),
                    )
                )
        efficiency_df["on_pareto_frontier"] = efficiency_df.apply(
            lambda row: (
                row.get("model_short"),
                row.get("condition"),
                row.get("judge_model"),
            )
            in frontier_keys
            or (row.get("model_short"), row.get("condition"), None) in frontier_keys,
            axis=1,
        )

    judge_rows = []
    for summary in summaries:
        overall = summary.get("overall") or {}
        exact = overall.get("exact_match") or {}
        llm_judged = overall.get("llm_judged") or {}
        totals = summary.get("totals") or {}
        judge_rows.append(
            {
                "judge_model": summary.get("judge_model"),
                "prompt_version": summary.get("prompt_version"),
                "evaluated": overall.get("total_evaluated"),
                "accuracy": overall.get("accuracy"),
                "correct": overall.get("correct"),
                "exact_match_rate": exact.get("rate"),
                "llm_judged_rate": llm_judged.get("rate"),
                "llm_approval_rate": llm_judged.get("approval_rate"),
                "accuracy_boost_from_llm": overall.get("accuracy_boost_from_llm"),
                "auto_exact": totals.get("auto_exact"),
                "llm_calls": totals.get("llm_calls"),
                "no_llm": totals.get("no_llm"),
                "no_llm_fallback_count": overall.get("no_llm_fallback_count"),
            }
        )
    judge_summary_df = pd.DataFrame(judge_rows)

    reliability_rows = []
    agreement_summary = agreement.get("summary") or {}
    for key in ["mean_pairwise_kappa", "min_pairwise_kappa", "max_pairwise_kappa"]:
        reliability_rows.append({"metric": key, "value": agreement_summary.get(key)})
    fleiss = agreement.get("fleiss_kappa") or {}
    alpha = agreement.get("krippendorff_alpha") or {}
    reliability_rows.extend(
        [
            {"metric": "fleiss_kappa", "value": fleiss.get("fleiss_kappa")},
            {"metric": "krippendorff_alpha", "value": alpha.get("alpha")},
            {"metric": "items_with_multiple_judges", "value": agreement.get("items_with_multiple_judges")},
            {"metric": "disagreement_count", "value": (agreement.get("disagreements") or {}).get("total_count")},
        ]
    )
    reliability_df = pd.DataFrame(reliability_rows).dropna(how="all")

    pairwise_rows = []
    for pair, values in (agreement.get("pairwise_cohen_kappa") or {}).items():
        pairwise_rows.append(
            {
                "judge_pair": pair,
                "cohen_kappa": values.get("kappa"),
                "agreement_rate": values.get("agreement_rate"),
                "shared_items": values.get("n_common"),
                "judge1_yes_rate": values.get("judge1_yes_rate"),
                "judge2_yes_rate": values.get("judge2_yes_rate"),
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)

    source_rows = []
    for source, values in (agreement.get("per_source_file") or {}).items():
        source_rows.append(
            {
                "source_file": source,
                "n_items": values.get("n_items"),
                "mean_kappa": values.get("mean_kappa"),
                "min_kappa": values.get("min_kappa"),
                "max_kappa": values.get("max_kappa"),
            }
        )
    source_agreement_df = pd.DataFrame(source_rows)

    disagreement_rows = []
    for item in (agreement.get("disagreements") or {}).get("sample", []):
        details = item.get("details") or {}
        disagreement_rows.append(
            {
                "source_file": details.get("source_file"),
                "ratings": item.get("ratings"),
                "input": details.get("input"),
                "gold": details.get("gold"),
                "prediction": details.get("prediction"),
            }
        )
    disagreement_df = pd.DataFrame(disagreement_rows)
    """

    combined_judge_cell = """
    import random

    BOOTSTRAP_ITERATIONS = 5000
    RANDOMIZATION_ITERATIONS = 5000
    RANDOM_SEED = 20260603
    TOP_COMPARISONS = 5


    def is_yes(value):
        return str(value).strip().lower() in {"yes", "true", "1", "y"}


    def item_identifier(item):
        if item.get("id") is not None:
            return str(item.get("id"))
        return json.dumps(
            {
                "input": item.get("input"),
                "gold": item.get("gold"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


    def load_combined_judge_items(eval_dir):
        evaluated_root = eval_dir / "evaluated_datasets"
        if not evaluated_root.exists():
            return pd.DataFrame()

        records = {}
        for path in sorted(evaluated_root.glob("*/*.json")):
            if path.name.startswith("summary"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            judge_model = payload.get("judge_model") or path.parent.name
            source_file = payload.get("source_file") or f"{path.stem.split('__judge-')[0]}.json"
            run_name = Path(source_file).stem
            model_short = payload.get("model_short")
            condition = payload.get("condition")
            seed = payload.get("seed")

            for item in payload.get("detailed_results", []):
                item_id = item_identifier(item)
                key = (run_name, seed, item_id)
                record = records.setdefault(
                    key,
                    {
                        "run_name": run_name,
                        "model_short": model_short,
                        "condition": condition,
                        "finetuned": payload.get("finetuned"),
                        "few_shot": payload.get("few_shot"),
                        "seed": seed,
                        "item_id": item_id,
                        "judge_votes": {},
                        "decision_methods": set(),
                    },
                )
                record["judge_votes"][judge_model] = is_yes(item.get("correct"))
                if item.get("decision_method"):
                    record["decision_methods"].add(item.get("decision_method"))

        rows = []
        for record in records.values():
            votes = list(record["judge_votes"].values())
            if not votes:
                continue
            num_judges = len(votes)
            yes_count = sum(1 for vote in votes if vote)
            rows.append(
                {
                    "run_name": record["run_name"],
                    "model_short": record["model_short"],
                    "condition": record["condition"],
                    "finetuned": record["finetuned"],
                    "few_shot": record["few_shot"],
                    "seed": record["seed"],
                    "item_id": record["item_id"],
                    "num_judges": num_judges,
                    "combined_score": yes_count / num_judges,
                    "consensus_score": float(yes_count == num_judges),
                    "any_judge_score": float(yes_count > 0),
                    "judge_disagreement": float(0 < yes_count < num_judges),
                    "exact_match_lower_bound": float("exact" in record["decision_methods"]),
                }
            )

        return pd.DataFrame(rows)


    def build_combined_ranking(item_df, fallback_df):
        if not item_df.empty:
            group_cols = ["model_short", "condition", "finetuned", "few_shot"]
            ranking = (
                item_df.groupby(group_cols, dropna=False)
                .agg(
                    n_items=("item_id", "count"),
                    num_seeds=("seed", "nunique"),
                    num_runs=("run_name", "nunique"),
                    mean_num_judges=("num_judges", "mean"),
                    combined_accuracy=("combined_score", "mean"),
                    consensus_accuracy=("consensus_score", "mean"),
                    any_judge_accuracy=("any_judge_score", "mean"),
                    exact_match_lower_bound=("exact_match_lower_bound", "mean"),
                    judge_disagreement_rate=("judge_disagreement", "mean"),
                )
                .reset_index()
            )
            return ranking.sort_values(
                ["combined_accuracy", "consensus_accuracy"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)

        if fallback_df.empty:
            return pd.DataFrame()

        fallback = (
            fallback_df.groupby(["model_short", "condition", "finetuned", "few_shot"], dropna=False)
            .agg(
                num_judges=("judge_model", "nunique"),
                num_seeds=("num_seeds", "max"),
                num_runs=("num_runs", "sum"),
                combined_accuracy=("accuracy", "mean"),
                exact_match_lower_bound=("exact_match_rate", "mean"),
                judge_accuracy_min=("accuracy", "min"),
                judge_accuracy_max=("accuracy", "max"),
                latency_mean_ms=("latency_mean_ms", "mean"),
            )
            .reset_index()
        )
        fallback["judge_accuracy_range"] = fallback["judge_accuracy_max"] - fallback["judge_accuracy_min"]
        return fallback.sort_values(
            ["combined_accuracy", "latency_mean_ms"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)


    def bootstrap_ci95(values, iterations=BOOTSTRAP_ITERATIONS, seed=RANDOM_SEED):
        clean_values = [float(value) for value in values if pd.notna(value)]
        if not clean_values:
            return None, None
        rng = random.Random(seed)
        sample_size = len(clean_values)
        means = []
        for _ in range(iterations):
            means.append(sum(rng.choice(clean_values) for _ in range(sample_size)) / sample_size)
        means.sort()
        low_idx = int(0.025 * (iterations - 1))
        high_idx = int(0.975 * (iterations - 1))
        return means[low_idx], means[high_idx]


    def paired_randomization_p_value(values, iterations=RANDOMIZATION_ITERATIONS, seed=RANDOM_SEED):
        clean_values = [float(value) for value in values if pd.notna(value)]
        if not clean_values:
            return None
        observed = abs(sum(clean_values) / len(clean_values))
        rng = random.Random(seed)
        extreme_count = 0
        for _ in range(iterations):
            randomized_mean = sum(
                value if rng.random() < 0.5 else -value for value in clean_values
            ) / len(clean_values)
            if abs(randomized_mean) >= observed - 1e-12:
                extreme_count += 1
        return (extreme_count + 1) / (iterations + 1)


    def build_paired_tests(item_df, ranking_df, top_k=TOP_COMPARISONS):
        if item_df.empty or ranking_df.empty:
            return pd.DataFrame()

        top_rows = ranking_df.head(top_k).to_dict("records")
        if len(top_rows) < 2:
            return pd.DataFrame()

        reference = top_rows[0]
        reference_scores = item_df[
            (item_df["model_short"] == reference["model_short"])
            & (item_df["condition"] == reference["condition"])
        ][["seed", "item_id", "combined_score"]].rename(
            columns={"combined_score": "reference_score"}
        )

        rows = []
        for candidate in top_rows[1:]:
            candidate_scores = item_df[
                (item_df["model_short"] == candidate["model_short"])
                & (item_df["condition"] == candidate["condition"])
            ][["seed", "item_id", "combined_score"]].rename(
                columns={"combined_score": "candidate_score"}
            )
            paired = reference_scores.merge(candidate_scores, on=["seed", "item_id"], how="inner")
            if paired.empty:
                continue

            differences = paired["reference_score"] - paired["candidate_score"]
            ci_low, ci_high = bootstrap_ci95(
                differences,
                seed=RANDOM_SEED + len(rows) + 1,
            )
            p_value = paired_randomization_p_value(
                differences,
                seed=RANDOM_SEED + 100 + len(rows),
            )
            rows.append(
                {
                    "comparison": (
                        f"{reference['model_short']} | {reference['condition']} vs "
                        f"{candidate['model_short']} | {candidate['condition']}"
                    ),
                    "n_pairs": len(paired),
                    "reference_accuracy": reference.get("combined_accuracy"),
                    "candidate_accuracy": candidate.get("combined_accuracy"),
                    "mean_difference": differences.mean(),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "paired_randomization_p": p_value,
                }
            )

        return pd.DataFrame(rows)


    combined_item_df = load_combined_judge_items(EVAL_DIR)
    combined_ranking_df = build_combined_ranking(combined_item_df, aggregate_df)
    if not combined_ranking_df.empty and "rank" not in combined_ranking_df.columns:
        combined_ranking_df.insert(0, "rank", range(1, len(combined_ranking_df) + 1))

    if combined_ranking_df.empty:
        print("No combined judge ranking data available.")
    else:
        combined_cols = [
            "rank",
            "model_short",
            "condition",
            "num_seeds",
            "num_runs",
            "n_items",
            "mean_num_judges",
            "combined_accuracy",
            "exact_match_lower_bound",
            "consensus_accuracy",
            "any_judge_accuracy",
            "judge_disagreement_rate",
            "judge_accuracy_range",
            "latency_mean_ms",
        ]
        print("Combined ranking: exact-match is a deterministic lower bound; non-exact correctness is averaged across judges.")
        display(combined_ranking_df[[c for c in combined_cols if c in combined_ranking_df.columns]].head(20).round(4))

    if aggregate_df.empty:
        print("No per-judge aggregate rows available for sensitivity analysis.")
    else:
        sensitivity_cols = [
            "model_short",
            "condition",
            "judge_model",
            "num_seeds",
            "num_runs",
            "accuracy",
            "accuracy_ci95",
            "exact_match_rate",
            "llm_approval_rate",
            "judge_agreement_kappa",
        ]
        per_judge_sensitivity_df = aggregate_df[[c for c in sensitivity_cols if c in aggregate_df.columns]].sort_values(
            ["judge_model", "accuracy"],
            ascending=[True, False],
            na_position="last",
        )
        print("Per-judge ranking sensitivity")
        display(per_judge_sensitivity_df.round(4))

        if "judge_model" in aggregate_df.columns and aggregate_df["judge_model"].nunique(dropna=True) > 1:
            judge_pivot_df = aggregate_df.pivot_table(
                index=["model_short", "condition"],
                columns="judge_model",
                values="accuracy",
                aggfunc="mean",
            )
            judge_pivot_df["judge_accuracy_range"] = judge_pivot_df.max(axis=1) - judge_pivot_df.min(axis=1)
            print("Largest per-judge accuracy gaps")
            display(
                judge_pivot_df.reset_index()
                .sort_values("judge_accuracy_range", ascending=False)
                .head(15)
                .round(4)
            )

    paired_tests_df = build_paired_tests(combined_item_df, combined_ranking_df)
    if paired_tests_df.empty:
        print("Paired bootstrap/randomization tests require item-level evaluated outputs for at least two model-condition groups.")
    else:
        print(
            "Paired top-model tests use shared seed/example pairs; "
            f"bootstrap={BOOTSTRAP_ITERATIONS}, randomization={RANDOMIZATION_ITERATIONS}."
        )
        display(paired_tests_df.round(4))
    """

    notebook = {
        "cells": [
            _markdown_cell(f"""
                # NL2ATL Publication Analysis

                Report directory: `{eval_dir.as_posix()}`

                This notebook consolidates the evidence needed for paper analysis: final judged accuracy, seed stability, judge reliability, accuracy-latency tradeoffs, and a compact reproducibility snapshot.
                """),
            _code_cell(load_cell),
            _code_cell(transform_cell),
            _markdown_cell("## Main Model Results"),
            _code_cell("""
                main_cols = [
                    "model_short",
                    "condition",
                    "judge_model",
                    "num_seeds",
                    "num_runs",
                    "accuracy",
                    "accuracy_ci95",
                    "exact_match_rate",
                    "llm_approval_rate",
                    "accuracy_boost_from_llm",
                    "latency_mean_ms",
                    "judge_agreement_kappa",
                ]

                if aggregate_df.empty:
                    print("No seed aggregate metrics were found.")
                else:
                    paper_results = aggregate_df[[c for c in main_cols if c in aggregate_df.columns]]
                    display(paper_results.round(4))

                    print("Best model per condition and judge")
                    best_by_condition = (
                        paper_results.sort_values("accuracy", ascending=False)
                        .groupby(["condition", "judge_model"], dropna=False)
                        .head(1)
                        .reset_index(drop=True)
                    )
                    display(best_by_condition.round(4))
                """),
            _markdown_cell("## Combined Judge Ranking And Sensitivity"),
            _code_cell(combined_judge_cell),
            _markdown_cell("## Seed Stability And Correctness Sources"),
            _code_cell("""
                if aggregate_df.empty:
                    print("No seed stability data available.")
                else:
                    stability_cols = [
                        "model_short",
                        "condition",
                        "judge_model",
                        "num_seeds",
                        "num_runs",
                        "accuracy",
                        "accuracy_std",
                        "accuracy_ci95",
                        "exact_match_rate",
                        "llm_approval_rate",
                        "accuracy_boost_from_llm",
                    ]
                    display(aggregate_df[[c for c in stability_cols if c in aggregate_df.columns]].round(4))

                    plot_df = aggregate_df.dropna(subset=["accuracy"]).head(20).iloc[::-1]
                    if not plot_df.empty:
                        labels = plot_df.apply(
                            lambda row: f"{row['model_short']} | {row['condition']} | {row['judge_model']}",
                            axis=1,
                        )
                        errors = plot_df["accuracy_ci95"] if "accuracy_ci95" in plot_df else None
                        plt.figure(figsize=(10, max(4, len(plot_df) * 0.35)))
                        plt.barh(labels, plot_df["accuracy"], xerr=errors, capsize=3)
                        plt.xlabel("Accuracy")
                        plt.title("Top model-condition groups with 95% seed interval")
                        plt.tight_layout()
                        plt.show()

                    contribution_cols = [
                        "model_short",
                        "condition",
                        "judge_model",
                        "accuracy_from_exact_match",
                        "accuracy_boost_from_llm",
                    ]
                    contribution_df = aggregate_df[[c for c in contribution_cols if c in aggregate_df.columns]].dropna(
                        subset=["accuracy_from_exact_match", "accuracy_boost_from_llm"],
                        how="all",
                    )
                    display(contribution_df.round(4))
                """),
            _markdown_cell("## Judge Reliability"),
            _code_cell("""
                if reliability_df.empty and judge_summary_df.empty:
                    print("No judge reliability data available.")
                else:
                    display(reliability_df.round(4))
                    if not judge_summary_df.empty:
                        display(judge_summary_df.round(4))
                    if not pairwise_df.empty:
                        display(pairwise_df.sort_values("cohen_kappa", ascending=False).round(4))
                    if not source_agreement_df.empty:
                        print("Lowest-agreement evaluated files")
                        display(
                            source_agreement_df.sort_values("mean_kappa", ascending=True)
                            .head(15)
                            .round(4)
                        )
                    if not disagreement_df.empty:
                        print("Sample judge disagreements for error analysis")
                        display(disagreement_df.head(10))
                """),
            _markdown_cell("## Accuracy-Latency Tradeoff"),
            _code_cell("""
                if efficiency_df.empty:
                    print("No accuracy-latency report was found.")
                else:
                    tradeoff_cols = [
                        "model_short",
                        "condition",
                        "judge_model",
                        "num_seeds",
                        "num_runs",
                        "accuracy",
                        "accuracy_std",
                        "latency_mean_ms",
                        "latency_p95_ms",
                        "throughput_samples_per_sec",
                        "on_pareto_frontier",
                    ]
                    display(
                        efficiency_df[[c for c in tradeoff_cols if c in efficiency_df.columns]]
                        .sort_values(["accuracy", "latency_mean_ms"], ascending=[False, True], na_position="last")
                        .round(4)
                    )
                    if not pareto_df.empty:
                        print("Pareto frontier")
                        display(pareto_df.round(4))

                    scatter_df = efficiency_df.dropna(subset=["accuracy", "latency_mean_ms"])
                    if not scatter_df.empty:
                        plt.figure(figsize=(9, 6))
                        non_frontier = scatter_df[~scatter_df["on_pareto_frontier"]]
                        frontier = scatter_df[scatter_df["on_pareto_frontier"]]
                        plt.scatter(non_frontier["latency_mean_ms"], non_frontier["accuracy"], alpha=0.55, label="Other")
                        plt.scatter(frontier["latency_mean_ms"], frontier["accuracy"], s=95, label="Pareto frontier")
                        for _, row in frontier.iterrows():
                            plt.annotate(
                                str(row.get("model_short", "")),
                                (row["latency_mean_ms"], row["accuracy"]),
                                fontsize=8,
                            )
                        plt.xlabel("Mean latency (ms)")
                        plt.ylabel("Accuracy")
                        plt.title("Accuracy vs latency")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                """),
            _markdown_cell("## Reproducibility Snapshot"),
            _code_cell("""
                if not manifest:
                    print("No reproducibility manifest was found.")
                else:
                    manifest_summary = pd.DataFrame(
                        [
                            {"field": "created_at", "value": manifest.get("created_at")},
                            {"field": "git_commit", "value": manifest.get("git_commit")},
                            {"field": "python_version", "value": manifest.get("python_version")},
                            {"field": "platform", "value": manifest.get("platform")},
                            {
                                "field": "prediction_files",
                                "value": len((manifest.get("inputs") or {}).get("prediction_files", [])),
                            },
                            {
                                "field": "evaluated_files",
                                "value": len((manifest.get("inputs") or {}).get("evaluated_files", [])),
                            },
                        ]
                    )
                    display(manifest_summary)

                    important_reports = {
                        "agreement_report.json",
                        "seed_aggregate_metrics_from_judged.json",
                        "efficiency_report.json",
                        "reproducibility_manifest.json",
                    }
                    report_rows = []
                    for item in manifest.get("reports", []):
                        name = Path(item.get("path", "")).name
                        if name in important_reports or name.startswith("summary__judge-"):
                            report_rows.append(
                                {
                                    "report": name,
                                    "sha256": item.get("sha256"),
                                    "bytes": item.get("bytes"),
                                }
                            )
                    display(pd.DataFrame(report_rows))
                """),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    save_json(notebook, output_path)


__all__ = ["DEFAULT_PUBLICATION_NOTEBOOK_NAME", "build_publication_notebook"]
