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

    combined_analysis_cell = """
    import hashlib


    def normalized_yes(value):
        return str(value).strip().lower() in {"yes", "true", "1", "y"}


    def comparison_item_key(item):
        if item.get("id") is not None:
            return str(item.get("id"))
        key_payload = {
            "input": item.get("input", ""),
            "gold": item.get("gold", ""),
            "gold_options": item.get("gold_options") or [],
        }
        key_text = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:16]


    def iter_evaluated_payloads(eval_dir):
        evaluated_dir = eval_dir / "evaluated_datasets"
        if not evaluated_dir.exists():
            return
        for judge_dir in sorted(path for path in evaluated_dir.iterdir() if path.is_dir()):
            judge_model = judge_dir.name
            for path in sorted(judge_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield judge_model, path, payload


    item_records = {}
    judge_records = []

    for judge_model, path, payload in iter_evaluated_payloads(EVAL_DIR):
        model_short = payload.get("model_short") or payload.get("model") or "unknown"
        condition = payload.get("condition") or "unknown"
        seed = payload.get("seed")

        for item in payload.get("detailed_results") or []:
            if not isinstance(item, dict):
                continue

            unit_key = f"{seed}|{comparison_item_key(item)}"
            record_key = (model_short, condition, seed, unit_key)
            correct_score = int(normalized_yes(item.get("correct")))
            decision_method = item.get("decision_method")
            is_exact = decision_method == "exact"

            record = item_records.setdefault(
                record_key,
                {
                    "model_short": model_short,
                    "condition": condition,
                    "seed": seed,
                    "unit_key": unit_key,
                    "judge_votes": {},
                    "decision_methods": {},
                    "is_exact": False,
                },
            )
            record["judge_votes"][judge_model] = correct_score
            record["decision_methods"][judge_model] = decision_method
            record["is_exact"] = record["is_exact"] or is_exact

            judge_records.append(
                {
                    "model_short": model_short,
                    "condition": condition,
                    "judge_model": judge_model,
                    "seed": seed,
                    "unit_key": unit_key,
                    "judge_accuracy": correct_score,
                    "exact_match_lower_bound": int(is_exact),
                }
            )

    combined_item_rows = []
    for record in item_records.values():
        votes = list(record["judge_votes"].values())
        if not votes:
            continue
        combined_item_rows.append(
            {
                "model_short": record["model_short"],
                "condition": record["condition"],
                "seed": record["seed"],
                "unit_key": record["unit_key"],
                "combined_score": sum(votes) / len(votes),
                "consensus_score": int(all(votes)),
                "any_judge_score": int(any(votes)),
                "judge_disagreement": int(len(set(votes)) > 1),
                "deterministic_exact_match_lower_bound": int(record["is_exact"]),
                "n_judges": len(votes),
            }
        )

    combined_item_df = pd.DataFrame(combined_item_rows)
    per_judge_item_df = pd.DataFrame(judge_records)

    if combined_item_df.empty:
        combined_ranking_df = pd.DataFrame()
    else:
        combined_ranking_df = (
            combined_item_df.groupby(["model_short", "condition"], dropna=False)
            .agg(
                combined_accuracy=("combined_score", "mean"),
                consensus_accuracy=("consensus_score", "mean"),
                any_judge_accuracy=("any_judge_score", "mean"),
                deterministic_exact_match_lower_bound=(
                    "deterministic_exact_match_lower_bound",
                    "mean",
                ),
                judge_disagreement_rate=("judge_disagreement", "mean"),
                mean_judges_per_item=("n_judges", "mean"),
                n_items=("combined_score", "size"),
                num_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        combined_ranking_df["accuracy_lift_over_exact_match"] = (
            combined_ranking_df["combined_accuracy"]
            - combined_ranking_df["deterministic_exact_match_lower_bound"]
        )

        if not aggregate_df.empty:
            latency_support_df = (
                aggregate_df.groupby(["model_short", "condition"], dropna=False)
                .agg(latency_mean_ms=("latency_mean_ms", "mean"))
                .reset_index()
            )
            combined_ranking_df = combined_ranking_df.merge(
                latency_support_df,
                on=["model_short", "condition"],
                how="left",
            )

        combined_ranking_df = combined_ranking_df.sort_values(
            ["combined_accuracy", "latency_mean_ms"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        combined_ranking_df.insert(0, "combined_rank", combined_ranking_df.index + 1)

    if per_judge_item_df.empty:
        per_judge_sensitivity_df = pd.DataFrame()
    else:
        per_judge_sensitivity_df = (
            per_judge_item_df.groupby(
                ["judge_model", "model_short", "condition"], dropna=False
            )
            .agg(
                judge_accuracy=("judge_accuracy", "mean"),
                exact_match_lower_bound=("exact_match_lower_bound", "mean"),
                n_items=("judge_accuracy", "size"),
                num_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        per_judge_sensitivity_df["judge_rank"] = per_judge_sensitivity_df.groupby(
            "judge_model"
        )["judge_accuracy"].rank(method="min", ascending=False)
        per_judge_sensitivity_df = per_judge_sensitivity_df.sort_values(
            ["judge_model", "judge_rank", "model_short", "condition"]
        ).reset_index(drop=True)

    if combined_ranking_df.empty:
        print("No detailed per-judge evaluated datasets were found for combined ranking.")
    else:
        print("Combined-judge ranking; each item receives 0, 0.5, or 1 from the two judge votes.")
        combined_cols = [
            "combined_rank",
            "model_short",
            "condition",
            "num_seeds",
            "n_items",
            "combined_accuracy",
            "deterministic_exact_match_lower_bound",
            "accuracy_lift_over_exact_match",
            "consensus_accuracy",
            "any_judge_accuracy",
            "judge_disagreement_rate",
            "latency_mean_ms",
        ]
        display(combined_ranking_df[[c for c in combined_cols if c in combined_ranking_df.columns]].round(4))

        print("Per-judge sensitivity analysis")
        sensitivity_cols = [
            "judge_model",
            "judge_rank",
            "model_short",
            "condition",
            "num_seeds",
            "n_items",
            "judge_accuracy",
            "exact_match_lower_bound",
        ]
        display(per_judge_sensitivity_df[[c for c in sensitivity_cols if c in per_judge_sensitivity_df.columns]].round(4))
    """

    paired_tests_cell = """
    import random


    def percentile(sorted_values, pct):
        if not sorted_values:
            return None
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * pct
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = position - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


    def paired_top_model_tests(item_df, ranking_df, top_k=6, n_iterations=2000, seed=20260604):
        if item_df.empty or ranking_df.empty:
            return pd.DataFrame()

        top_groups = list(
            ranking_df.head(top_k)[["model_short", "condition"]].itertuples(
                index=False, name=None
            )
        )
        if len(top_groups) < 2:
            return pd.DataFrame()

        best_group = top_groups[0]
        best_scores = (
            item_df[
                (item_df["model_short"] == best_group[0])
                & (item_df["condition"] == best_group[1])
            ]
            .set_index("unit_key")["combined_score"]
            .rename("best_score")
        )

        comparison_rows = []
        for comparison_index, challenger_group in enumerate(top_groups[1:], start=1):
            challenger_scores = (
                item_df[
                    (item_df["model_short"] == challenger_group[0])
                    & (item_df["condition"] == challenger_group[1])
                ]
                .set_index("unit_key")["combined_score"]
                .rename("challenger_score")
            )
            paired = best_scores.to_frame().join(challenger_scores, how="inner").dropna()
            if paired.empty:
                continue

            differences = (paired["best_score"] - paired["challenger_score"]).tolist()
            n_items = len(differences)
            observed_difference = sum(differences) / n_items
            rng = random.Random(seed + comparison_index * 1009)

            bootstrap_means = []
            for _ in range(n_iterations):
                total = 0.0
                for _ in range(n_items):
                    total += differences[rng.randrange(n_items)]
                bootstrap_means.append(total / n_items)
            bootstrap_means.sort()

            nonzero_differences = [value for value in differences if value != 0]
            if nonzero_differences:
                extreme_count = 0
                observed_abs = abs(observed_difference)
                for _ in range(n_iterations):
                    total = 0.0
                    for value in nonzero_differences:
                        total += value if rng.random() < 0.5 else -value
                    randomized_mean = total / n_items
                    if abs(randomized_mean) >= observed_abs - 1e-12:
                        extreme_count += 1
                randomization_p_value = (extreme_count + 1) / (n_iterations + 1)
            else:
                randomization_p_value = 1.0

            comparison_rows.append(
                {
                    "best_model": best_group[0],
                    "best_condition": best_group[1],
                    "challenger_model": challenger_group[0],
                    "challenger_condition": challenger_group[1],
                    "n_paired_items": n_items,
                    "observed_accuracy_difference": observed_difference,
                    "bootstrap_ci95_low": percentile(bootstrap_means, 0.025),
                    "bootstrap_ci95_high": percentile(bootstrap_means, 0.975),
                    "randomization_p_value": randomization_p_value,
                    "iterations": n_iterations,
                }
            )

        return pd.DataFrame(comparison_rows)


    paired_test_df = paired_top_model_tests(combined_item_df, combined_ranking_df)
    if paired_test_df.empty:
        print("No paired top-model comparisons were available.")
    else:
        print("Paired comparisons use shared seed-item units and combined judge scores.")
        display(paired_test_df.round(4))
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
            _markdown_cell("## Combined-Judge Ranking"),
            _code_cell(combined_analysis_cell),
            _markdown_cell("## Paired Top-Model Tests"),
            _code_cell(paired_tests_cell),
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
