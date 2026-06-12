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
    """Write one focused notebook for AAAI-oriented result analysis.

    The notebook is intentionally curated to the evidence reviewers expect:
    experimental scale, headline accuracy by model and condition, the
    exact-match vs LLM-judge decomposition, statistical significance of the
    leading system, judge reliability validated against humans, the
    accuracy-latency tradeoff, and a compact reproducibility snapshot.
    """
    eval_dir = Path(eval_dir)
    configured_eval_dir = json.dumps(eval_dir.resolve().as_posix())

    load_cell = """
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    # Defaults tuned for figures that drop straight into a paper.
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

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
        f"Loaded {len(aggregates)} model-condition groups "
        f"and {len(summaries)} judge summaries."
    )
    """.replace("__CONFIGURED_EVAL_DIR__", configured_eval_dir)

    transform_cell = """
    # Logical ordering for conditions so figures read zero-shot -> few-shot -> finetuned.
    KNOWN_CONDITION_ORDER = {
        "baseline_zero_shot": 0,
        "zero_shot": 0,
        "baseline_few_shot": 1,
        "few_shot": 1,
        "finetuned": 2,
        "finetuned_zero_shot": 2,
        "finetuned_few_shot": 3,
    }


    def condition_sort_key(condition):
        return (KNOWN_CONDITION_ORDER.get(condition, 50), str(condition))


    def metric_field(metrics, name, stat="mean"):
        value = (metrics or {}).get(name)
        if isinstance(value, dict):
            return value.get(stat)
        return value if stat == "mean" else None


    def judge_label(item):
        if item.get("judge_model"):
            return item.get("judge_model")
        judge_models = item.get("judge_models") or []
        return ", ".join(str(value) for value in judge_models) or "unknown"


    def agreement_mean(item, name):
        value = (item.get("judge_agreement") or {}).get(name)
        if isinstance(value, dict):
            return value.get("mean")
        return value


    def kappa_label(value):
        if value is None:
            return "n/a"
        if value < 0:
            return "poor"
        if value <= 0.20:
            return "slight"
        if value <= 0.40:
            return "fair"
        if value <= 0.60:
            return "moderate"
        if value <= 0.80:
            return "substantial"
        return "almost perfect"


    aggregate_rows = []
    for item in aggregates:
        metrics = item.get("metrics") or {}
        exact_base = metric_field(metrics, "accuracy_from_exact_match")
        if exact_base is None:
            exact_base = metric_field(metrics, "exact_match_rate")
        aggregate_rows.append(
            {
                "model_short": item.get("model_short"),
                "condition": item.get("condition"),
                "judge_model": judge_label(item),
                "split_type": item.get("split_type"),
                "num_seeds": item.get("num_seeds"),
                "num_folds": item.get("num_folds"),
                "num_runs": item.get("num_runs"),
                "n_examples": metric_field(metrics, "n_examples"),
                "accuracy": metric_field(metrics, "accuracy"),
                "accuracy_std": metric_field(metrics, "accuracy", "std"),
                "accuracy_ci95": metric_field(metrics, "accuracy", "ci95"),
                "exact_match_rate": metric_field(metrics, "exact_match_rate"),
                "llm_approval_rate": metric_field(metrics, "llm_approval_rate"),
                "accuracy_from_exact_match": exact_base,
                "accuracy_boost_from_llm": metric_field(metrics, "accuracy_boost_from_llm"),
                "latency_mean_ms": metric_field(metrics, "latency_mean_ms"),
                "latency_p95_ms": metric_field(metrics, "latency_p95_ms"),
                "judge_agreement_kappa": agreement_mean(item, "confidence_score"),
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
                "exact_match_rate": exact.get("rate"),
                "llm_approval_rate": llm_judged.get("approval_rate"),
                "accuracy_boost_from_llm": overall.get("accuracy_boost_from_llm"),
                "llm_calls": totals.get("llm_calls"),
            }
        )
    judge_summary_df = pd.DataFrame(judge_rows)
    """

    setup_cell = """
    def unique_sorted(values):
        return sorted({value for value in values if value is not None}, key=str)


    setup_records = []
    if not aggregate_df.empty:
        models = unique_sorted(aggregate_df["model_short"])
        conditions = sorted(unique_sorted(aggregate_df["condition"]), key=condition_sort_key)
        n_examples_values = [v for v in aggregate_df["n_examples"].tolist() if pd.notna(v)]
        seed_values = [v for v in aggregate_df["num_seeds"].tolist() if pd.notna(v)]
        fold_values = [v for v in aggregate_df["num_folds"].tolist() if pd.notna(v)]
        split_types = unique_sorted(aggregate_df["split_type"])
        setup_records.extend(
            [
                {"property": "Models evaluated", "value": f"{len(models)}: {', '.join(models)}"},
                {"property": "Conditions", "value": f"{len(conditions)}: {', '.join(conditions)}"},
                {
                    "property": "Test examples per run",
                    "value": int(max(n_examples_values)) if n_examples_values else "n/a",
                },
                {
                    "property": "Seeds per canonical model",
                    "value": int(max(seed_values)) if seed_values else "n/a",
                },
                {
                    "property": "Cross-validation folds",
                    "value": int(max(fold_values)) if fold_values and max(fold_values) else 0,
                },
                {"property": "Split protocols", "value": ", ".join(split_types) or "n/a"},
            ]
        )

    judges = agreement.get("judges")
    if not judges and not aggregate_df.empty:
        judges = unique_sorted(aggregate_df["judge_model"])
    judges = judges or []
    prompt_versions = unique_sorted(summary.get("prompt_version") for summary in summaries)
    setup_records.extend(
        [
            {
                "property": "LLM judges",
                "value": f"{len(judges)}: {', '.join(map(str, judges))}" if judges else "n/a",
            },
            {
                "property": "Judge prompt version",
                "value": ", ".join(prompt_versions) if prompt_versions else "n/a",
            },
            {
                "property": "Items rated by 2+ judges",
                "value": agreement.get("items_with_multiple_judges", "n/a"),
            },
        ]
    )

    setup_df = pd.DataFrame(setup_records)
    if setup_df.empty:
        print("No experiment metadata was found.")
    else:
        display(setup_df)
    """

    main_results_cell = """
    PRIMARY_COLUMNS = [
        "model_short",
        "condition",
        "judge_model",
        "num_seeds",
        "accuracy",
        "accuracy_ci95",
        "exact_match_rate",
        "llm_approval_rate",
        "latency_mean_ms",
        "judge_agreement_kappa",
    ]

    if aggregate_df.empty:
        print("No seed aggregate metrics were found.")
    else:
        results_table = aggregate_df[[c for c in PRIMARY_COLUMNS if c in aggregate_df.columns]]
        print("Final judged accuracy per model, condition, and judge (mean over seeds).")
        display(results_table.round(4))

        best_rows = (
            aggregate_df.sort_values("accuracy", ascending=False, na_position="last")
            .groupby(["condition", "judge_model"], dropna=False)
            .head(1)
            .sort_values("accuracy", ascending=False)
            .reset_index(drop=True)
        )
        print("Best model per condition and judge.")
        display(best_rows[[c for c in PRIMARY_COLUMNS if c in best_rows.columns]].round(4))

        # Headline figure: accuracy by model and condition, averaged across judges.
        headline = (
            aggregate_df.groupby(["model_short", "condition"], dropna=False)
            .agg(accuracy=("accuracy", "mean"), ci95=("accuracy_ci95", "mean"))
            .reset_index()
            .dropna(subset=["accuracy"])
        )
        if not headline.empty:
            accuracy_pivot = headline.pivot(index="model_short", columns="condition", values="accuracy")
            error_pivot = headline.pivot(index="model_short", columns="condition", values="ci95")
            ordered_conditions = sorted(accuracy_pivot.columns, key=condition_sort_key)
            accuracy_pivot = accuracy_pivot[ordered_conditions]
            error_pivot = error_pivot[ordered_conditions].fillna(0)
            axis = accuracy_pivot.plot(
                kind="bar",
                yerr=error_pivot,
                capsize=3,
                figsize=(max(6, len(accuracy_pivot) * 1.6), 5),
            )
            axis.set_ylabel("Accuracy")
            axis.set_xlabel("Model")
            axis.set_ylim(0, 1)
            axis.set_title("Accuracy by model and condition (mean across judges, 95% seed CI)")
            axis.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.show()
    """

    decomposition_cell = """
    print("Each judged accuracy is deterministic exact matches plus the LLM-judge rescue.")

    if aggregate_df.empty:
        print("No accuracy decomposition data was found.")
    else:
        decomposition = (
            aggregate_df.groupby(["model_short", "condition"], dropna=False)
            .agg(
                exact_match=("accuracy_from_exact_match", "mean"),
                llm_boost=("accuracy_boost_from_llm", "mean"),
                accuracy=("accuracy", "mean"),
            )
            .reset_index()
            .dropna(subset=["exact_match", "llm_boost"], how="all")
        )
        if decomposition.empty:
            print("No exact-match / LLM-judge breakdown was available.")
        else:
            decomposition = decomposition.sort_values(
                "accuracy", ascending=False, na_position="last"
            )
            decomposition_table = decomposition.assign(
                llm_boost_share=(
                    decomposition["llm_boost"] / decomposition["accuracy"]
                ).where(decomposition["accuracy"] > 0)
            )
            display(decomposition_table.round(4))

            plot_df = decomposition.copy()
            plot_df["label"] = plot_df["model_short"] + " | " + plot_df["condition"]
            plot_df = plot_df.set_index("label")[["exact_match", "llm_boost"]].fillna(0).iloc[::-1]
            axis = plot_df.plot(
                kind="barh",
                stacked=True,
                figsize=(9, max(3, len(plot_df) * 0.5)),
            )
            axis.set_xlabel("Accuracy contribution")
            axis.set_xlim(0, 1)
            axis.set_title("Exact match vs LLM-judge contribution to accuracy")
            axis.legend(
                ["Exact match", "LLM-judge rescue"],
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
            plt.tight_layout()
            plt.show()
    """

    significance_cell = """
    import hashlib
    import random


    def normalized_yes(value):
        return str(value).strip().lower() in {"yes", "true", "1", "y"}


    def comparison_item_key(item):
        if item.get("id") is not None:
            return str(item.get("id"))
        payload = {
            "input": item.get("input", ""),
            "gold": item.get("gold", ""),
            "gold_options": item.get("gold_options") or [],
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


    def iter_evaluated_payloads(eval_dir):
        evaluated_dir = eval_dir / "evaluated_datasets"
        if not evaluated_dir.exists():
            return
        for judge_dir in sorted(path for path in evaluated_dir.iterdir() if path.is_dir()):
            for path in sorted(judge_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield judge_dir.name, payload


    # Each item gets a combined judge score in {0, 0.5, 1} (the mean of the judge votes).
    item_votes = {}
    for judge_model, payload in iter_evaluated_payloads(EVAL_DIR):
        model_short = payload.get("model_short") or payload.get("model") or "unknown"
        condition = payload.get("condition") or "unknown"
        seed = payload.get("seed")
        for item in payload.get("detailed_results") or []:
            if not isinstance(item, dict):
                continue
            unit_key = f"{seed}|{comparison_item_key(item)}"
            votes = item_votes.setdefault((model_short, condition, unit_key), [])
            votes.append(int(normalized_yes(item.get("correct"))))

    combined_item_df = pd.DataFrame(
        [
            {
                "model_short": model_short,
                "condition": condition,
                "unit_key": unit_key,
                "combined_score": sum(votes) / len(votes),
            }
            for (model_short, condition, unit_key), votes in item_votes.items()
            if votes
        ]
    )


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


    def paired_top_model_tests(item_df, ranked_groups, n_iterations=2000, seed=20260604):
        if item_df.empty or len(ranked_groups) < 2:
            return pd.DataFrame()

        best_group = ranked_groups[0]
        best_scores = (
            item_df[
                (item_df["model_short"] == best_group[0])
                & (item_df["condition"] == best_group[1])
            ]
            .set_index("unit_key")["combined_score"]
            .rename("best_score")
        )

        rows = []
        for index, challenger in enumerate(ranked_groups[1:], start=1):
            challenger_scores = (
                item_df[
                    (item_df["model_short"] == challenger[0])
                    & (item_df["condition"] == challenger[1])
                ]
                .set_index("unit_key")["combined_score"]
                .rename("challenger_score")
            )
            paired = best_scores.to_frame().join(challenger_scores, how="inner").dropna()
            if paired.empty:
                continue

            differences = (paired["best_score"] - paired["challenger_score"]).tolist()
            n_items = len(differences)
            observed = sum(differences) / n_items
            rng = random.Random(seed + index * 1009)

            bootstrap_means = []
            for _ in range(n_iterations):
                total = 0.0
                for _ in range(n_items):
                    total += differences[rng.randrange(n_items)]
                bootstrap_means.append(total / n_items)
            bootstrap_means.sort()

            nonzero = [value for value in differences if value != 0]
            if nonzero:
                observed_abs = abs(observed)
                extreme = 0
                for _ in range(n_iterations):
                    total = 0.0
                    for value in nonzero:
                        total += value if rng.random() < 0.5 else -value
                    if abs(total / n_items) >= observed_abs - 1e-12:
                        extreme += 1
                p_value = (extreme + 1) / (n_iterations + 1)
            else:
                p_value = 1.0

            rows.append(
                {
                    "best_model": best_group[0],
                    "best_condition": best_group[1],
                    "challenger_model": challenger[0],
                    "challenger_condition": challenger[1],
                    "n_paired_items": n_items,
                    "accuracy_difference": observed,
                    "ci95_low": percentile(bootstrap_means, 0.025),
                    "ci95_high": percentile(bootstrap_means, 0.975),
                    "randomization_p_value": p_value,
                }
            )
        return pd.DataFrame(rows)


    if combined_item_df.empty or aggregate_df.empty:
        print("No per-judge evaluated datasets were found for paired significance testing.")
    else:
        ranked_groups = list(
            aggregate_df.groupby(["model_short", "condition"], dropna=False)["accuracy"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        paired_test_df = paired_top_model_tests(combined_item_df, ranked_groups[:6])
        if paired_test_df.empty:
            print("Not enough shared items to compare the leading models.")
        else:
            print(
                "Best model-condition vs the next-best groups on shared seed-item units "
                "(combined judge score); bootstrap CI and a randomization p-value."
            )
            display(paired_test_df.round(4))
    """

    reliability_cell = """
    reliability_rows = []
    summary_block = agreement.get("summary") or {}
    for key in ["mean_pairwise_kappa", "min_pairwise_kappa", "max_pairwise_kappa"]:
        value = summary_block.get(key)
        reliability_rows.append(
            {"metric": key, "value": value, "interpretation": kappa_label(value)}
        )
    fleiss_value = (agreement.get("fleiss_kappa") or {}).get("fleiss_kappa")
    alpha_value = (agreement.get("krippendorff_alpha") or {}).get("alpha")
    reliability_rows.append(
        {"metric": "fleiss_kappa", "value": fleiss_value, "interpretation": kappa_label(fleiss_value)}
    )
    reliability_rows.append(
        {"metric": "krippendorff_alpha", "value": alpha_value, "interpretation": kappa_label(alpha_value)}
    )
    reliability_df = pd.DataFrame(reliability_rows).dropna(subset=["value"])

    breakdown = agreement.get("agreement_breakdown") or {}
    breakdown_summary = breakdown.get("agreement_summary") or {}
    full_agreement = breakdown_summary.get("full_agreement")
    total_items = breakdown.get("total_items")
    if full_agreement is not None and total_items:
        print(
            f"Judges rendered identical verdicts on {full_agreement}/{total_items} items "
            f"({full_agreement / total_items:.1%}); unanimous across all judges on "
            f"{breakdown.get('unanimous_all_judges', 0)}."
        )

    if reliability_df.empty:
        print("No inter-rater reliability metrics were found.")
    else:
        print(
            f"Inter-rater reliability over "
            f"{agreement.get('items_with_multiple_judges', 'n/a')} shared items."
        )
        display(reliability_df.round(4))

    pairwise_rows = []
    for pair, values in (agreement.get("pairwise_cohen_kappa") or {}).items():
        kappa = values.get("kappa")
        pairwise_rows.append(
            {
                "judge_pair": pair,
                "cohen_kappa": kappa,
                "interpretation": kappa_label(kappa),
                "agreement_rate": values.get("agreement_rate"),
                "shared_items": values.get("n_common"),
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        display(pairwise_df.sort_values("cohen_kappa", ascending=False, na_position="last").round(4))

    human = agreement.get("human_comparison")
    if human:
        human_rows = []
        for judge, values in (human.get("per_judge") or {}).items():
            human_rows.append(
                {
                    "judge_model": judge,
                    "accuracy_vs_human": values.get("accuracy"),
                    "shared_items": values.get("n_common"),
                }
            )
        majority = human.get("majority_vote") or {}
        unanimous = human.get("unanimous") or {}
        if majority.get("accuracy") is not None:
            human_rows.append(
                {
                    "judge_model": "majority_vote",
                    "accuracy_vs_human": majority.get("accuracy"),
                    "shared_items": majority.get("n_items"),
                }
            )
        if unanimous.get("accuracy") is not None:
            human_rows.append(
                {
                    "judge_model": "unanimous_judges",
                    "accuracy_vs_human": unanimous.get("accuracy"),
                    "shared_items": unanimous.get("n_items"),
                }
            )
        human_df = pd.DataFrame(human_rows)
        if not human_df.empty:
            print("Agreement of LLM judges with human annotations (validation of the judge).")
            display(human_df.round(4))
            human_plot = human_df.dropna(subset=["accuracy_vs_human"])
            if not human_plot.empty:
                plt.figure(figsize=(7, max(3, len(human_plot) * 0.5)))
                plt.barh(human_plot["judge_model"], human_plot["accuracy_vs_human"])
                plt.xlabel("Agreement with human labels")
                plt.xlim(0, 1)
                plt.title("LLM-judge agreement with human annotations")
                plt.tight_layout()
                plt.show()
    else:
        print("No human-validation comparison was found in the agreement report.")
        kappa_plot = pairwise_df.dropna(subset=["cohen_kappa"]) if not pairwise_df.empty else pairwise_df
        if not kappa_plot.empty:
            plt.figure(figsize=(7, max(3, len(kappa_plot) * 0.5)))
            plt.barh(kappa_plot["judge_pair"], kappa_plot["cohen_kappa"])
            plt.xlabel("Cohen's kappa")
            plt.xlim(0, 1)
            plt.title("Pairwise judge agreement (Cohen's kappa)")
            plt.tight_layout()
            plt.show()

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
    if not disagreement_df.empty:
        print("Sample judge disagreements for qualitative error analysis.")
        display(disagreement_df.head(5))
    """

    tradeoff_cell = """
    TRADEOFF_COLUMNS = [
        "model_short",
        "condition",
        "judge_model",
        "num_seeds",
        "accuracy",
        "accuracy_std",
        "latency_mean_ms",
        "latency_p95_ms",
        "on_pareto_frontier",
    ]

    if efficiency_df.empty:
        print("No accuracy-latency report was found.")
    else:
        tradeoff_table = efficiency_df[
            [c for c in TRADEOFF_COLUMNS if c in efficiency_df.columns]
        ].sort_values(
            ["accuracy", "latency_mean_ms"], ascending=[False, True], na_position="last"
        )
        display(tradeoff_table.round(4))
        if not pareto_df.empty:
            print("Pareto-optimal groups (no other group is both faster and more accurate).")
            display(pareto_df.round(4))

        scatter_df = efficiency_df.dropna(subset=["accuracy", "latency_mean_ms"])
        if not scatter_df.empty:
            plt.figure(figsize=(9, 6))
            if "on_pareto_frontier" in scatter_df.columns:
                others = scatter_df[~scatter_df["on_pareto_frontier"]]
                frontier = scatter_df[scatter_df["on_pareto_frontier"]]
            else:
                others = scatter_df
                frontier = scatter_df.iloc[0:0]
            plt.scatter(others["latency_mean_ms"], others["accuracy"], alpha=0.55, label="Other")
            if not frontier.empty:
                frontier_sorted = frontier.sort_values("latency_mean_ms")
                plt.scatter(
                    frontier_sorted["latency_mean_ms"],
                    frontier_sorted["accuracy"],
                    s=95,
                    label="Pareto frontier",
                )
                plt.plot(
                    frontier_sorted["latency_mean_ms"],
                    frontier_sorted["accuracy"],
                    linestyle="--",
                    alpha=0.6,
                )
                for _, row in frontier_sorted.iterrows():
                    plt.annotate(
                        str(row.get("model_short", "")),
                        (row["latency_mean_ms"], row["accuracy"]),
                        fontsize=8,
                    )
            plt.xlabel("Mean latency per example (ms)")
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs latency tradeoff")
            plt.legend()
            plt.tight_layout()
            plt.show()
    """

    reproducibility_cell = """
    if not manifest:
        print("No reproducibility manifest was found.")
    else:
        inputs = manifest.get("inputs") or {}
        rows = [
            {"field": "created_at", "value": manifest.get("created_at")},
            {"field": "git_commit", "value": manifest.get("git_commit")},
            {"field": "python_version", "value": manifest.get("python_version")},
            {"field": "platform", "value": manifest.get("platform")},
            {"field": "prediction_files", "value": len(inputs.get("prediction_files", []))},
            {"field": "evaluated_files", "value": len(inputs.get("evaluated_files", []))},
            {"field": "report_files", "value": len(manifest.get("reports", []))},
        ]
        display(pd.DataFrame(rows))
        for note in manifest.get("limitations", []) or []:
            print(f"- {note}")
    """

    notebook = {
        "cells": [
            _markdown_cell(f"""
                # NL2ATL Publication Analysis

                Report directory: `{eval_dir.as_posix()}`

                This notebook curates the evidence an AAAI reviewer expects: experimental scale, headline accuracy by model and condition, where the accuracy comes from (exact match vs LLM-judge rescue), the statistical significance of the leading system, judge reliability validated against human labels, the accuracy-latency tradeoff, and a compact reproducibility snapshot.
                """),
            _code_cell(load_cell),
            _code_cell(transform_cell),
            _markdown_cell("""
                ## Experimental Setup

                Scale and protocol of the evaluation in one glance.
                """),
            _code_cell(setup_cell),
            _markdown_cell("""
                ## Main Results

                Final judged accuracy with 95% seed confidence intervals, plus the headline figure for the paper.
                """),
            _code_cell(main_results_cell),
            _markdown_cell("""
                ## Accuracy Decomposition: Exact Match vs LLM Judge

                How much accuracy is deterministic exact match and how much is recovered by the LLM judge.
                """),
            _code_cell(decomposition_cell),
            _markdown_cell("""
                ## Statistical Significance

                Is the best system significantly better than its closest competitors? Paired bootstrap and randomization tests on shared items.
                """),
            _code_cell(significance_cell),
            _markdown_cell("""
                ## Judge Reliability and Human Validation

                Inter-rater agreement (with interpretation) and, when available, how closely the LLM judges track human annotations.
                """),
            _code_cell(reliability_cell),
            _markdown_cell("""
                ## Accuracy-Latency Tradeoff

                Pareto-optimal systems for deployment decisions.
                """),
            _code_cell(tradeoff_cell),
            _markdown_cell("""
                ## Reproducibility

                Provenance snapshot for the reported numbers.
                """),
            _code_cell(reproducibility_cell),
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
