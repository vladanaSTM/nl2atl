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
    """Deprecated: Import from src.evaluation.difficulty instead."""

    from src.evaluation.difficulty import *  # noqa
    import warnings

    warnings.warn(
        "Importing from src.classify_difficulty is deprecated. "
        "Use src.evaluation.difficulty instead.",
        DeprecationWarning,
        stacklevel=2,
    )
                and formula_body[i - 1] != "_"
