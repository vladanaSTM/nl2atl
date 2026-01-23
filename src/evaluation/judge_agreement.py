"""
Inter-rater agreement metrics (Cohen's Kappa, Fleiss' Kappa) for LLM judges.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..infra.io import load_json, save_json


def create_item_key(item: dict) -> str:
    """Create a unique key for an (input, gold, prediction) tuple."""
    key_data = json.dumps(
        {
            "input": item.get("input", ""),
            "gold": item.get("gold", ""),
            "prediction": item.get("prediction", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def load_evaluated_files(eval_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """
    Load all evaluated datasets organized by judge.

    Returns:
        {judge_name: {source_file_stem: [items]}}
    """
    results = {}

    for judge_dir in eval_dir.iterdir():
        if not judge_dir.is_dir():
            continue

        judge_name = judge_dir.name
        results[judge_name] = {}

        for json_file in judge_dir.glob("*.json"):
            if json_file.name.startswith("summary"):
                continue

            try:
                data = load_json(json_file)
                if isinstance(data, list):
                    source_file = json_file.stem.split("__judge-")[0]
                    results[judge_name][source_file] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    return results


def align_judgments(
    judge_results: Dict[str, Dict[str, List[dict]]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, dict]]:
    """
    Align judgments across judges for the same items.

    Returns:
        aligned: {item_key: {judge_name: "yes"|"no"}}
        item_details: {item_key: {input, gold, prediction, source_file}}
    """
    aligned = defaultdict(dict)
    item_details = {}

    for judge_name, source_files in judge_results.items():
        for source_file, items in source_files.items():
            for item in items:
                item_key = f"{source_file}::{create_item_key(item)}"
                decision = item.get("correct", "no").lower()
                if decision not in {"yes", "no"}:
                    decision = "no"
                aligned[item_key][judge_name] = decision

                if item_key not in item_details:
                    item_details[item_key] = {
                        "input": item.get("input", ""),
                        "gold": item.get("gold", ""),
                        "prediction": item.get("prediction", ""),
                        "source_file": source_file,
                        "decision_method": item.get("decision_method", "unknown"),
                    }

    return dict(aligned), item_details


def filter_common_items(
    aligned: Dict[str, Dict[str, str]],
    min_judges: int = 2,
) -> Dict[str, Dict[str, str]]:
    """Keep only items rated by at least min_judges judges."""
    return {key: judges for key, judges in aligned.items() if len(judges) >= min_judges}


def compute_cohen_kappa(labels1: List[str], labels2: List[str]) -> float:
    """Compute Cohen's Kappa between two raters."""
    n = len(labels1)
    if n == 0:
        return 0.0

    # Observed agreement
    agreements = sum(1 for a, b in zip(labels1, labels2) if a == b)
    po = agreements / n

    # Label frequencies
    categories = ["yes", "no"]
    freq1 = {c: labels1.count(c) / n for c in categories}
    freq2 = {c: labels2.count(c) / n for c in categories}

    # Expected agreement by chance
    pe = sum(freq1[c] * freq2[c] for c in categories)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for multiple raters.

    Args:
        ratings_matrix: (n_items, n_categories) matrix where each cell
                       contains the count of raters who assigned that category
    """
    n_items, n_categories = ratings_matrix.shape
    n_raters_per_item = ratings_matrix.sum(axis=1)

    if n_items == 0 or n_raters_per_item.max() <= 1:
        return 1.0

    # Proportion of ratings in each category
    total_ratings = ratings_matrix.sum()
    p_j = ratings_matrix.sum(axis=0) / total_ratings

    # Observed agreement per item
    P_i = np.zeros(n_items)
    for i in range(n_items):
        n_i = n_raters_per_item[i]
        if n_i > 1:
            P_i[i] = (np.sum(ratings_matrix[i] ** 2) - n_i) / (n_i * (n_i - 1))
        else:
            P_i[i] = 1.0

    P_o = P_i.mean()
    P_e = np.sum(p_j**2)

    if P_e == 1.0:
        return 1.0 if P_o == 1.0 else 0.0

    return (P_o - P_e) / (1 - P_e)


def compute_pairwise_kappa(
    aligned: Dict[str, Dict[str, str]],
    judges: List[str],
) -> Dict[Tuple[str, str], dict]:
    """Compute Cohen's Kappa for each pair of judges."""
    results = {}

    for judge1, judge2 in combinations(judges, 2):
        common_items = [
            key
            for key, ratings in aligned.items()
            if judge1 in ratings and judge2 in ratings
        ]

        if not common_items:
            results[(judge1, judge2)] = {
                "kappa": None,
                "n_common": 0,
                "agreement_rate": None,
            }
            continue

        labels1 = [aligned[key][judge1] for key in common_items]
        labels2 = [aligned[key][judge2] for key in common_items]

        kappa = compute_cohen_kappa(labels1, labels2)
        agreement = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(labels1)

        # Confusion matrix
        tp = sum(1 for a, b in zip(labels1, labels2) if a == "yes" and b == "yes")
        tn = sum(1 for a, b in zip(labels1, labels2) if a == "no" and b == "no")
        fn = sum(1 for a, b in zip(labels1, labels2) if a == "yes" and b == "no")
        fp = sum(1 for a, b in zip(labels1, labels2) if a == "no" and b == "yes")

        results[(judge1, judge2)] = {
            "kappa": round(kappa, 4),
            "n_common": len(common_items),
            "agreement_rate": round(agreement, 4),
            "confusion_matrix": {
                "both_yes": tp,
                "both_no": tn,
                f"{judge1}_yes_{judge2}_no": fn,
                f"{judge1}_no_{judge2}_yes": fp,
            },
            "judge1_yes_rate": round(labels1.count("yes") / len(labels1), 4),
            "judge2_yes_rate": round(labels2.count("yes") / len(labels2), 4),
        }

    return results


def compute_overall_fleiss(
    aligned: Dict[str, Dict[str, str]],
    judges: List[str],
) -> dict:
    """Compute Fleiss' Kappa across all judges."""
    complete_items = [
        key for key, ratings in aligned.items() if all(j in ratings for j in judges)
    ]

    if not complete_items:
        return {"fleiss_kappa": None, "n_items": 0, "n_judges": len(judges)}

    categories = ["yes", "no"]
    matrix = np.zeros((len(complete_items), 2))

    for i, key in enumerate(complete_items):
        for judge in judges:
            decision = aligned[key][judge]
            cat_idx = categories.index(decision)
            matrix[i, cat_idx] += 1

    kappa = compute_fleiss_kappa(matrix)

    return {
        "fleiss_kappa": round(kappa, 4),
        "n_items": len(complete_items),
        "n_judges": len(judges),
    }


def find_disagreements(
    aligned: Dict[str, Dict[str, str]],
    item_details: Dict[str, dict],
    judges: List[str],
) -> List[dict]:
    """Find items where judges disagree."""
    disagreements = []

    for item_key, ratings in aligned.items():
        if len(ratings) < 2:
            continue

        decisions = list(ratings.values())
        if len(set(decisions)) > 1:
            disagreements.append(
                {
                    "item_key": item_key,
                    "ratings": ratings,
                    "details": item_details.get(item_key, {}),
                }
            )

    return disagreements


def compute_per_source_agreement(
    aligned: Dict[str, Dict[str, str]],
    judges: List[str],
) -> Dict[str, dict]:
    """Compute agreement metrics per source file."""
    source_items = defaultdict(dict)
    for key, ratings in aligned.items():
        source = key.split("::")[0]
        source_items[source][key] = ratings

    per_source = {}
    for source, items in sorted(source_items.items()):
        if not items:
            continue

        pairwise = compute_pairwise_kappa(items, judges)
        kappas = [v["kappa"] for v in pairwise.values() if v["kappa"] is not None]

        per_source[source] = {
            "n_items": len(items),
            "mean_kappa": round(np.mean(kappas), 4) if kappas else None,
            "min_kappa": round(np.min(kappas), 4) if kappas else None,
            "max_kappa": round(np.max(kappas), 4) if kappas else None,
            "pairwise": {
                f"{j1}_vs_{j2}": v["kappa"] for (j1, j2), v in pairwise.items()
            },
        }

    return per_source


def compute_agreement_breakdown(
    aligned: Dict[str, Dict[str, str]],
    judges: List[str],
) -> dict:
    """Compute detailed breakdown of agreement patterns."""
    n_judges = len(judges)

    by_rater_count = defaultdict(list)
    for item_key, ratings in aligned.items():
        by_rater_count[len(ratings)].append((item_key, ratings))

    breakdown = {
        "total_items": len(aligned),
        "by_rater_coverage": {},
        "agreement_summary": {
            "full_agreement": 0,
            "partial_disagreement": 0,
        },
        "unanimous_all_judges": 0,
    }

    for n_raters, items in sorted(by_rater_count.items()):
        full_agree = 0
        disagree = 0

        for item_key, ratings in items:
            decisions = list(ratings.values())
            if len(set(decisions)) == 1:
                full_agree += 1
                breakdown["agreement_summary"]["full_agreement"] += 1
                if n_raters == n_judges:
                    breakdown["unanimous_all_judges"] += 1
            else:
                disagree += 1
                breakdown["agreement_summary"]["partial_disagreement"] += 1

        all_judges_present = set().union(*[set(r.keys()) for _, r in items])
        breakdown["by_rater_coverage"][f"{n_raters}_raters"] = {
            "count": len(items),
            "judges_present": sorted(all_judges_present),
            "full_agreement": full_agree,
            "any_disagreement": disagree,
        }

    return breakdown


def compute_detailed_disagreement_stats(
    aligned: Dict[str, Dict[str, str]],
    judges: List[str],
) -> dict:
    """Compute detailed statistics about disagreement patterns."""
    disagreement_patterns = defaultdict(int)

    for item_key, ratings in aligned.items():
        if len(ratings) < 2:
            continue

        decisions = list(ratings.values())
        if len(set(decisions)) == 1:
            pattern = "unanimous"
        else:
            yes_count = decisions.count("yes")
            no_count = decisions.count("no")
            total = len(decisions)

            if yes_count > no_count:
                pattern = f"majority_yes_{yes_count}_of_{total}"
            elif no_count > yes_count:
                pattern = f"majority_no_{no_count}_of_{total}"
            else:
                pattern = f"split_{yes_count}_vs_{no_count}"

        disagreement_patterns[pattern] += 1

    return {
        "patterns": dict(disagreement_patterns),
        "unanimous_count": disagreement_patterns.get("unanimous", 0),
        "any_disagreement_count": sum(
            v for k, v in disagreement_patterns.items() if k != "unanimous"
        ),
    }


def generate_agreement_report(
    eval_dir: Path,
    output_path: Optional[Path] = None,
    judges: Optional[List[str]] = None,
    include_disagreements: bool = True,
    max_disagreements: int = 50,
) -> dict:
    """Generate a complete inter-rater agreement report."""
    judge_results = load_evaluated_files(eval_dir)

    if judges:
        missing = [j for j in judges if j not in judge_results]
        if missing:
            raise ValueError(f"Requested judges not found: {missing}")
        judge_results = {j: judge_results[j] for j in judges}

    if len(judge_results) < 2:
        raise ValueError(
            f"Need at least 2 judges for agreement analysis. Found: {list(judge_results.keys())}"
        )

    judges = sorted(judge_results.keys())
    print(f"Found {len(judges)} judges: {judges}")

    aligned, item_details = align_judgments(judge_results)
    common_aligned = filter_common_items(aligned, min_judges=2)

    print(f"Total unique items: {len(aligned)}")
    print(f"Items rated by 2+ judges: {len(common_aligned)}")

    pairwise = compute_pairwise_kappa(common_aligned, judges)
    fleiss_result = compute_overall_fleiss(common_aligned, judges)
    per_source = compute_per_source_agreement(common_aligned, judges)
    agreement_breakdown = compute_agreement_breakdown(common_aligned, judges)
    disagreement_stats = compute_detailed_disagreement_stats(common_aligned, judges)

    all_disagreements = find_disagreements(common_aligned, item_details, judges)
    disagreements = []
    if include_disagreements:
        disagreements = all_disagreements[:max_disagreements]

    kappa_values = [v["kappa"] for v in pairwise.values() if v["kappa"] is not None]
    summary = {}
    if kappa_values:
        summary = {
            "mean_pairwise_kappa": round(float(np.mean(kappa_values)), 4),
            "min_pairwise_kappa": round(float(np.min(kappa_values)), 4),
            "max_pairwise_kappa": round(float(np.max(kappa_values)), 4),
            "std_pairwise_kappa": round(float(np.std(kappa_values)), 4),
        }

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "judges": judges,
        "n_judges": len(judges),
        "total_unique_items": len(aligned),
        "items_with_multiple_judges": len(common_aligned),
        "summary": summary,
        "fleiss_kappa": fleiss_result,
        "agreement_breakdown": agreement_breakdown,
        "disagreement_stats": disagreement_stats,
        "pairwise_cohen_kappa": {
            f"{j1}_vs_{j2}": data for (j1, j2), data in pairwise.items()
        },
        "per_source_file": per_source,
        "disagreements": {
            "total_count": len(all_disagreements),
            "sample": disagreements,
        },
        "interpretation": {
            "kappa_scale": {
                "< 0.00": "Poor (less than chance)",
                "0.00 - 0.20": "Slight agreement",
                "0.21 - 0.40": "Fair agreement",
                "0.41 - 0.60": "Moderate agreement",
                "0.61 - 0.80": "Substantial agreement",
                "0.81 - 1.00": "Almost perfect agreement",
            }
        },
    }

    if output_path:
        save_json(report, output_path)
        print(f"Agreement report saved to: {output_path}")

    return report


def print_agreement_summary(report: dict) -> None:
    """Print a formatted summary of the agreement report."""
    print("\n" + "=" * 70)
    print("INTER-RATER AGREEMENT REPORT")
    print("=" * 70)

    print(f"\nJudges: {', '.join(report['judges'])}")
    print(f"Items rated by 2+ judges: {report['items_with_multiple_judges']}")

    print("\n--- Pairwise Cohen's Kappa ---")
    for pair, data in report["pairwise_cohen_kappa"].items():
        kappa = data.get("kappa")
        n = data.get("n_common", 0)
        agreement = data.get("agreement_rate")
        if kappa is not None:
            interp = _interpret_kappa(kappa)
            print(
                f"  {pair}: κ = {kappa:.3f} ({interp}, n={n}, raw agreement={agreement:.1%})"
            )
        else:
            print(f"  {pair}: No common items")

    fleiss = report.get("fleiss_kappa", {})
    if fleiss.get("fleiss_kappa") is not None:
        print(f"\n--- Fleiss' Kappa (all {fleiss['n_judges']} judges) ---")
        print(f"  κ = {fleiss['fleiss_kappa']:.3f} (n={fleiss['n_items']} items)")

    if report.get("summary"):
        s = report["summary"]
        print("\n--- Summary Statistics ---")
        print(f"  Mean pairwise κ:  {s['mean_pairwise_kappa']:.3f}")
        print(f"  Std pairwise κ:   {s['std_pairwise_kappa']:.3f}")
        print(
            f"  Range: [{s['min_pairwise_kappa']:.3f}, {s['max_pairwise_kappa']:.3f}]"
        )

    disagreements = report.get("disagreements", {})
    if disagreements.get("total_count", 0) > 0:
        print(f"\n--- Disagreements ---")
        print(f"  Total: {disagreements['total_count']} items with judge disagreement")

    print("\n--- Kappa Interpretation Guide ---")
    for range_str, meaning in report["interpretation"]["kappa_scale"].items():
        print(f"  {range_str}: {meaning}")

    print("=" * 70)


def _interpret_kappa(kappa: float) -> str:
    """Interpret kappa value."""
    if kappa < 0:
        return "Poor"
    elif kappa < 0.21:
        return "Slight"
    elif kappa < 0.41:
        return "Fair"
    elif kappa < 0.61:
        return "Moderate"
    elif kappa < 0.81:
        return "Substantial"
    else:
        return "Almost Perfect"


compute_agreement_metrics = generate_agreement_report
