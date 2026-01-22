#!/usr/bin/env python3
"""
Difficulty Classification Algorithm for NL-to-ATL Translation Dataset

Two-dimensional classification based on:
1. Formula Complexity: Syntactic complexity of the ATL output formula
2. NL Ambiguity: Linguistic ambiguities in the natural language input

Output: "easy" or "hard"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Data path
DATA = "data/dataset.json"


# ============================================================================
# FORMULA COMPLEXITY ANALYSIS
# ============================================================================


def extract_coalition(formula: str) -> List[str]:
    """Extract agents from coalition <<Agent1, Agent2, ...>>"""
    match = re.search(r"<<([^>]+)>>", formula)
    if match:
        return [a.strip() for a in match.group(1).split(",")]
    return []


def count_temporal_operators(formula: str) -> Dict[str, int]:
    """Count temporal operators in formula."""
    formula_body = re.sub(r"<<[^>]+>>", "", formula)

    return {
        "G": len(re.findall(r"\bG\b", formula_body)),
        "F": len(re.findall(r"\bF\b", formula_body)),
        "X": len(re.findall(r"\bX\b", formula_body)),
        "U": len(re.findall(r"\bU\b", formula_body)),
        "W": len(re.findall(r"\bW\b", formula_body)),
        "R": len(re.findall(r"\bR\b", formula_body)),
    }


def count_logical_connectives(formula: str) -> int:
    """Count logical connectives."""
    formula_body = re.sub(r"<<[^>]+>>", "", formula)

    count = 0
    count += len(re.findall(r"->", formula_body))
    count += len(re.findall(r"→", formula_body))
    count += len(re.findall(r"&&", formula_body))
    count += len(re.findall(r"∧", formula_body))
    count += len(re.findall(r"\|\|", formula_body))
    count += len(re.findall(r"∨", formula_body))
    count += len(re.findall(r"!(?!=)", formula_body))
    count += len(re.findall(r"¬", formula_body))

    return count


def calculate_nesting_depth(formula: str) -> int:
    """Calculate maximum nesting depth of temporal operators."""
    formula_body = re.sub(r"<<[^>]+>>", "", formula)
    temporal_ops = {"G", "F", "X", "U", "W", "R"}

    max_depth = 0
    paren_depth = 0
    temp_depths = []

    i = 0
    while i < len(formula_body):
        char = formula_body[i]

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char in temporal_ops:
            before_ok = (
                i == 0
                or not formula_body[i - 1].isalnum()
                and formula_body[i - 1] != "_"
            )
            after_ok = (
                i == len(formula_body) - 1
                or not formula_body[i + 1].isalnum()
                and formula_body[i + 1] != "_"
            )
            if before_ok and after_ok:
                temp_depths.append(paren_depth)

        i += 1

    if temp_depths:
        max_depth = max(temp_depths) + 1
        # Bonus for multiple operators at different depths
        if len(set(temp_depths)) > 1:
            max_depth += 1

    return max_depth


def formula_complexity_score(formula: str) -> float:
    """
    Calculate formula complexity score.

    Components:
    - Temporal operator count (weight: 1.0 each)
    - Nesting depth (weight: 1.5 per level)
    - Logical connectives (weight: 0.5 each)
    - Coalition size (weight: 0.5 per additional agent)
    """
    score = 0.0

    # Temporal operators
    temp_ops = count_temporal_operators(formula)
    score += sum(temp_ops.values()) * 1.0

    # Nesting depth
    score += calculate_nesting_depth(formula) * 1.5

    # Logical connectives
    score += count_logical_connectives(formula) * 0.5

    # Coalition size
    coalition = extract_coalition(formula)
    if len(coalition) > 1:
        score += (len(coalition) - 1) * 0.5

    return score


# ============================================================================
# NATURAL LANGUAGE AMBIGUITY ANALYSIS
# ============================================================================


def detect_scope_ambiguity(nl_input: str) -> float:
    """Detect potential scope ambiguities."""
    score = 0.0
    nl_lower = nl_input.lower()

    patterns = [
        (r"if .+ at the next step", 2.0),
        (r"at the next step .+ if", 2.0),
        (r"always .+ if .+ then", 1.5),
        (r"if .+ then .+ always", 1.5),
        (r"eventually .+ if", 1.5),
        (r"never .+ if", 1.5),
        (r"at some point .+ if", 1.5),
    ]

    for pattern, weight in patterns:
        if re.search(pattern, nl_lower):
            score += weight

    return score


def detect_implicit_operators(nl_input: str, formula: str) -> float:
    """Detect operators in formula not explicitly mentioned in NL."""
    score = 0.0
    nl_lower = nl_input.lower()
    temp_ops = count_temporal_operators(formula)

    # Implicit G
    if temp_ops.get("G", 0) > 0:
        g_keywords = [
            "always",
            "forever",
            "permanently",
            "continuously",
            "at all times",
            "invariably",
            "globally",
        ]
        if not any(kw in nl_lower for kw in g_keywords):
            if "never" in nl_lower:
                score += 0.5  # "never" implies G but is somewhat explicit
            elif "if" in nl_lower:
                score += 1.0  # Implicit G for conditionals
            else:
                score += 1.5

    # Implicit F
    if temp_ops.get("F", 0) > 0:
        f_keywords = [
            "eventually",
            "sooner or later",
            "finally",
            "at some point",
            "someday",
            "in the future",
        ]
        if not any(kw in nl_lower for kw in f_keywords):
            score += 1.5

    # Implicit X
    if temp_ops.get("X", 0) > 0:
        x_keywords = [
            "next step",
            "next state",
            "immediately after",
            "in the next",
            "at the next",
        ]
        if not any(kw in nl_lower for kw in x_keywords):
            score += 1.5

    # Implicit U
    if temp_ops.get("U", 0) > 0:
        u_keywords = ["until", "up to"]
        if not any(kw in nl_lower for kw in u_keywords):
            score += 2.0

    return score


def detect_semantic_gap(nl_input: str, formula: str) -> float:
    """Detect semantic gaps between NL and formula."""
    score = 0.0
    nl_lower = nl_input.lower()

    gap_patterns = [
        (r"move away from", 1.5),
        (r"leave the", 1.5),
        (r"exit the", 1.5),
        (r"depart from", 1.5),
        (r"no longer", 1.0),
        (r"stop being", 1.0),
        (r"cease to", 1.0),
        (r"become unavailable", 1.0),
        (r"fail to", 1.0),
        (r"avoid", 1.0),
        (r"prevent", 1.0),
        (r"maintain", 0.5),
        (r"keep", 0.5),
        (r"pass through", 1.0),
        (r"reach", 0.5),
    ]

    for pattern, weight in gap_patterns:
        if re.search(pattern, nl_lower):
            score += weight

    # Negation mismatch
    formula_body = re.sub(r"<<[^>]+>>", "", formula)
    has_formula_neg = "!" in formula_body or "¬" in formula_body

    neg_words = ["not", "n't", "never", "without"]
    has_nl_neg = any(neg in nl_lower for neg in neg_words)

    if has_formula_neg and not has_nl_neg:
        score += 1.0

    return score


def detect_pronoun_complexity(nl_input: str) -> float:
    """Detect pronoun resolution complexity."""
    score = 0.0
    pronouns = ["they", "them", "their", "it", "its"]

    for pronoun in pronouns:
        count = len(re.findall(r"\b" + pronoun + r"\b", nl_input.lower()))
        score += count * 0.5

    return score


def detect_temporal_ambiguity(nl_input: str) -> float:
    """Detect ambiguous temporal expressions."""
    score = 0.0
    nl_lower = nl_input.lower()

    patterns = [
        (r"\bafter\b", 1.5),
        (r"\bbefore\b", 2.0),
        (r"\bwhile\b", 1.5),
        (r"\bduring\b", 1.5),
        (r"\bwhen\b", 1.0),
        (r"\bonce\b", 1.5),
        (r"\bas soon as\b", 1.5),
        (r"\bunless\b", 1.5),
        (r"\bwhenever\b", 1.0),
    ]

    for pattern, weight in patterns:
        if re.search(pattern, nl_lower):
            score += weight

    return score


def detect_agent_complexity(nl_input: str, formula: str) -> float:
    """Detect complexity in agent references."""
    score = 0.0
    nl_lower = nl_input.lower()
    coalition = extract_coalition(formula)

    if len(coalition) > 2:
        score += 0.5 * (len(coalition) - 2)

    patterns = [
        (r"together", 0.5),
        (r"jointly", 0.5),
        (r"cooperat", 0.5),
        (r"both .+ and", 0.5),
        (r"either .+ or", 1.0),
        (r"number \d+", 0.3),
    ]

    for pattern, weight in patterns:
        if re.search(pattern, nl_lower):
            score += weight

    return score


def detect_structural_complexity(nl_input: str) -> float:
    """Detect complex sentence structures."""
    score = 0.0
    nl_lower = nl_input.lower()

    # Multiple conditionals
    if_count = len(re.findall(r"\bif\b", nl_lower))
    if if_count > 1:
        score += (if_count - 1) * 1.5

    # Multiple clauses
    for marker in [", and ", ", or ", ", but ", ", then "]:
        score += nl_lower.count(marker) * 0.3

    # Sentence length
    word_count = len(nl_input.split())
    if word_count > 20:
        score += (word_count - 20) * 0.05

    return score


def nl_ambiguity_score(nl_input: str, formula: str) -> float:
    """Calculate total NL ambiguity score."""
    return (
        detect_scope_ambiguity(nl_input)
        + detect_implicit_operators(nl_input, formula)
        + detect_semantic_gap(nl_input, formula)
        + detect_pronoun_complexity(nl_input)
        + detect_temporal_ambiguity(nl_input)
        + detect_agent_complexity(nl_input, formula)
        + detect_structural_complexity(nl_input)
    )


# ============================================================================
# CLASSIFICATION
# ============================================================================


def classify_difficulty(
    nl_input: str,
    formula: str,
    formula_weight: float = 0.4,
    nl_weight: float = 0.6,
    threshold: float = 5.0,
) -> Tuple[str, Dict[str, float]]:
    """
    Classify difficulty as 'easy' or 'hard'.

    Returns: (classification, detailed_scores)
    """
    fc = formula_complexity_score(formula)
    na = nl_ambiguity_score(nl_input, formula)
    combined = formula_weight * fc + nl_weight * na

    classification = "hard" if combined > threshold else "easy"

    return classification, {
        "formula_complexity": round(fc, 2),
        "nl_ambiguity": round(na, 2),
        "combined_score": round(combined, 2),
        "threshold": threshold,
    }


def process_dataset(
    input_path: Union[Path, str] = DATA,
    output_path: Optional[Union[Path, str]] = None,
    formula_weight: float = 0.4,
    nl_weight: float = 0.6,
    threshold: float = 5.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Process dataset and update difficulty labels."""

    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    stats = {"easy": 0, "hard": 0}
    all_scores = []

    for item in dataset:
        nl_input = item.get("input", "")
        formula = item.get("output", "")

        classification, scores = classify_difficulty(
            nl_input,
            formula,
            formula_weight=formula_weight,
            nl_weight=nl_weight,
            threshold=threshold,
        )

        item["difficulty"] = classification
        all_scores.append(scores["combined_score"])

        if verbose:
            item["_debug_scores"] = scores

        stats[classification] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 60)
    print("DIFFICULTY CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(dataset)}")
    print(f"  Easy: {stats['easy']:4d} ({100*stats['easy']/len(dataset):5.1f}%)")
    print(f"  Hard: {stats['hard']:4d} ({100*stats['hard']/len(dataset):5.1f}%)")
    print("-" * 60)
    print("Score statistics:")
    print(f"  Min:    {min(all_scores):5.2f}")
    print(f"  Max:    {max(all_scores):5.2f}")
    print(f"  Mean:   {sum(all_scores)/len(all_scores):5.2f}")
    print(f"  Threshold: {threshold}")
    print("-" * 60)
    print(f"Saved to: {output_path}")
    print("=" * 60)

    return dataset


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify NL-to-ATL translation difficulty"
    )
    parser.add_argument("--input", "-i", type=Path, default=DATA)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--formula-weight", "-fw", type=float, default=0.4)
    parser.add_argument("--nl-weight", "-nw", type=float, default=0.6)
    parser.add_argument("--threshold", "-t", type=float, default=5.0)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        formula_weight=args.formula_weight,
        nl_weight=args.nl_weight,
        threshold=args.threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
