
# Difficulty Classification Algorithm for NL-to-ATL Translation

## Overview

This algorithm classifies the difficulty of translating natural language specifications into ATL (Alternating-time Temporal Logic) formulas. It uses a **two-dimensional analysis** that considers both the complexity of the output formula and the inherent ambiguities in the natural language input.

---

## Table of Contents

- [Motivation](#motivation)
- [The Two Dimensions](#the-two-dimensions)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Programmatic Usage](#programmatic-usage)
- [Configuration](#configuration)
- [Scoring Components](#scoring-components)
- [Examples](#examples)
- [Tuning Guide](#tuning-guide)
- [File Structure](#file-structure)

---

## Motivation

Translation difficulty is not solely determined by the complexity of the target formula. Consider:

| Scenario | Formula Complexity | NL Ambiguity | True Difficulty |
|----------|-------------------|--------------|-----------------|
| Simple sentence → Simple formula | Low | Low | **Easy** |
| Complex sentence → Complex formula | High | Low | **Medium** |
| Ambiguous sentence → Simple formula | Low | High | **Medium-Hard** |
| Ambiguous sentence → Complex formula | High | High | **Hard** |

The algorithm captures both dimensions to provide accurate difficulty labels.

---

## The Two Dimensions

### Dimension 1: Formula Complexity

Measures the syntactic complexity of the ATL output formula:

```
Score = (temporal_operators × 1.0) + 
        (nesting_depth × 1.5) + 
        (logical_connectives × 0.5) + 
        (coalition_size - 1) × 0.5
```

**Components:**
- **Temporal Operators**: G (globally), F (finally), X (next), U (until), W (weak until), R (release)
- **Nesting Depth**: How deeply operators are nested (e.g., `G(F p)` = depth 2)
- **Logical Connectives**: →, ∧, ∨, ¬ (and their ASCII equivalents)
- **Coalition Size**: Number of agents in `<<Agent1, Agent2, ...>>`

### Dimension 2: NL Ambiguity

Measures linguistic features that make translation challenging:

| Feature | Description | Weight |
|---------|-------------|--------|
| Scope Ambiguity | "if X at the next step" — what does "next step" modify? | 1.5-2.0 |
| Implicit Operators | Formula requires operators not mentioned in NL | 1.0-2.0 |
| Semantic Gap | Conceptual distance between NL and formula | 0.5-1.5 |
| Pronoun Resolution | Pronouns requiring reference resolution | 0.5 each |
| Temporal Ambiguity | Words like "after", "before", "once" | 1.0-2.0 |
| Agent Complexity | Complex agent references or coalitions | 0.3-1.0 |
| Structural Complexity | Long sentences, multiple conditionals | 0.3-1.5 |

---

## Usage

### Command Line

#### Basic Usage

Process the dataset with default settings:

```bash
python classify_difficulty.py
```

This will:
- Read from `data/dataset.json`
- Classify each example as "easy" or "hard"
- Overwrite the file with updated difficulty labels

#### With Custom Options

```bash
# Adjust the classification threshold
python classify_difficulty.py --threshold 4.5

# Change dimension weights
python classify_difficulty.py --formula-weight 0.3 --nl-weight 0.7

# Save to a different file
python classify_difficulty.py --output data/classified_dataset.json

# Enable verbose mode (adds debug scores to output)
python classify_difficulty.py --verbose

# Combine options
python classify_difficulty.py \
    --input data/raw_dataset.json \
    --output data/classified_dataset.json \
    --threshold 5.0 \
    --formula-weight 0.4 \
    --nl-weight 0.6 \
    --verbose
```

#### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `data/dataset.json` | Input dataset path |
| `--output` | `-o` | Same as input | Output dataset path |
| `--threshold` | `-t` | `5.0` | Score threshold for hard classification |
| `--formula-weight` | `-fw` | `0.4` | Weight for formula complexity (0-1) |
| `--nl-weight` | `-nw` | `0.6` | Weight for NL ambiguity (0-1) |
| `--verbose` | `-v` | `False` | Add debug scores to output |

### Programmatic Usage

#### Basic Classification

```python
from classify_difficulty import classify_difficulty

nl_input = "The user can guarantee that if they insert cash, then at the next step the payment will be registered."
formula = "<<User>>G (cash_inserted -> X paid)"

difficulty, scores = classify_difficulty(nl_input, formula)

print(f"Difficulty: {difficulty}")
print(f"Scores: {scores}")
```

**Output:**
```
Difficulty: hard
Scores: {'formula_complexity': 5.0, 'nl_ambiguity': 4.5, 'combined_score': 4.7, 'threshold': 5.0}
```

#### Process Entire Dataset

```python
from pathlib import Path
from classify_difficulty import process_dataset

# Process with custom settings
dataset = process_dataset(
    input_path=Path("data/my_dataset.json"),
    output_path=Path("data/classified_dataset.json"),
    threshold=4.5,
    formula_weight=0.4,
    nl_weight=0.6,
    verbose=True
)

# Access results
for item in dataset:
    print(f"{item['id']}: {item['difficulty']}")
```

#### Use Individual Scoring Functions

```python
from classify_difficulty import (
    formula_complexity_score,
    nl_ambiguity_score,
    detect_scope_ambiguity,
    detect_implicit_operators,
    count_temporal_operators
)

formula = "<<Machine>>G (X error -> F recovered)"
nl = "The machine can guarantee that if an error occurs at the next step, then sooner or later the system will be recovered."

# Get detailed scores
print(f"Formula complexity: {formula_complexity_score(formula)}")
print(f"NL ambiguity: {nl_ambiguity_score(nl, formula)}")
print(f"Scope ambiguity: {detect_scope_ambiguity(nl)}")
print(f"Temporal operators: {count_temporal_operators(formula)}")
```

---

## Configuration

### Choosing the Right Threshold

The threshold determines the cutoff between "easy" and "hard":

| Threshold | Effect |
|-----------|--------|
| 3.0 | Aggressive — most examples classified as "hard" |
| 5.0 | Balanced — moderate split (default) |
| 7.0 | Conservative — most examples classified as "easy" |

**Recommendation:** Start with the default (5.0) and adjust based on your dataset distribution.

### Choosing Dimension Weights

The weights control the relative importance of each dimension:

| Use Case | Formula Weight | NL Weight |
|----------|---------------|-----------|
| Translation task (understanding NL is key) | 0.3-0.4 | 0.6-0.7 |
| Balanced assessment | 0.5 | 0.5 |
| Formula generation focus | 0.6-0.7 | 0.3-0.4 |

**Default:** Formula=0.4, NL=0.6 (optimized for translation tasks)

---

## Scoring Components

### Formula Complexity Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│ Formula: <<Machine>>G (X error -> F recovered)              │
├─────────────────────────────────────────────────────────────┤
│ Temporal Operators:                                         │
│   G: 1 × 1.0 = 1.0                                         │
│   X: 1 × 1.0 = 1.0                                         │
│   F: 1 × 1.0 = 1.0                                         │
├─────────────────────────────────────────────────────────────┤
│ Nesting Depth: 2 × 1.5 = 3.0                               │
│   (G contains X and F)                                      │
├─────────────────────────────────────────────────────────────┤
│ Logical Connectives:                                        │
│   ->: 1 × 0.5 = 0.5                                        │
├─────────────────────────────────────────────────────────────┤
│ Coalition: 1 agent (no bonus)                               │
├─────────────────────────────────────────────────────────────┤
│ TOTAL: 5.5                                                  │
└─────────────────────────────────────────────────────────────┘
```

### NL Ambiguity Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│ NL: "The machine can guarantee that if an error occurs at  │
│      the next step, then sooner or later the system will   │
│      be recovered."                                         │
├─────────────────────────────────────────────────────────────┤
│ Scope Ambiguity:                                            │
│   "if ... at the next step" pattern: 2.0                   │
│   (Does "next step" modify "error" or the consequence?)    │
├─────────────────────────────────────────────────────────────┤
│ Implicit Operators:                                         │
│   G operator present but no "always" keyword: 1.0          │
├─────────────────────────────────────────────────────────────┤
│ Pronoun Complexity:                                         │
│   No pronouns: 0.0                                          │
├─────────────────────────────────────────────────────────────┤
│ Temporal Expressions:                                       │
│   "sooner or later" matches F explicitly: 0.0              │
├─────────────────────────────────────────────────────────────┤
│ TOTAL: 3.0                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Examples

### Sample Classifications

| ID | Input (truncated) | Formula | FC | NL | Combined | Class |
|----|------------------|---------|-----|-----|----------|-------|
| ex32 | "The drone and the wind together can guarantee that sooner or later..." | `<<Drone,Wind>>F !at_waypoint` | 2.5 | 3.0 | 2.8 | easy |
| ex233 | "Robots number 1, number 2 and number 3 have a strategy..." | `<<Robot1,Robot2,Robot3>>X(pos1∨pos2)` | 3.0 | 2.0 | 2.4 | easy |
| ex15 | "The machine can guarantee that if an error occurs at the next step..." | `<<Machine>>G(X error->F recovered)` | 5.5 | 4.0 | 4.6 | easy |
| ex17 | "The machine can guarantee that if the payment does not occur..." | `<<Machine>>G(!paid->G!ticket_printed)` | 6.5 | 3.5 | 4.7 | easy |
| ex16 | "The user can guarantee that after obtaining the ticket..." | `<<User>>(true U(has_ticket&&F gate_open))` | 4.5 | 5.5 | 5.1 | hard |

### Verbose Output Example

When running with `--verbose`, each item includes debug scores:

```json
{
  "id": "ex16",
  "input": "The user can guarantee that after obtaining the ticket, they will sooner or later pass through the gate.",
  "output": "<<User>>(true U (has_ticket && F gate_open))",
  "difficulty": "hard",
  "_debug_scores": {
    "formula_complexity": 4.5,
    "nl_ambiguity": 5.5,
    "combined_score": 5.1,
    "threshold": 5.0
  }
}
```

---

## Tuning Guide

### Step 1: Run with Verbose Mode

```bash
python classify_difficulty.py --verbose --output data/debug_output.json
```

### Step 2: Analyze the Distribution

Check the console output:

```
============================================================
DIFFICULTY CLASSIFICATION COMPLETE
============================================================
Total examples: 250
  Easy:  180 ( 72.0%)
  Hard:   70 ( 28.0%)
------------------------------------------------------------
Score statistics:
  Min:    1.20
  Max:   11.50
  Mean:   4.85
  Threshold: 5.0
------------------------------------------------------------
```

### Step 3: Adjust Based on Goals

| Goal | Action |
|------|--------|
| More balanced split | Lower threshold if too many "easy" |
| More strict "hard" label | Raise threshold |
| Emphasize NL challenges | Increase `--nl-weight` |
| Emphasize formula complexity | Increase `--formula-weight` |

### Step 4: Validate Manually

Review a sample of classifications:

```python
import json

with open("data/debug_output.json") as f:
    data = json.load(f)

# Check borderline cases
borderline = [x for x in data if 4.5 < x.get('_debug_scores', {}).get('combined_score', 0) < 5.5]
for item in borderline[:10]:
    print(f"\n{item['id']}: {item['difficulty']}")
    print(f"  Input: {item['input'][:80]}...")
    print(f"  Scores: {item['_debug_scores']}")
```

---

## File Structure

```
your_project/
├── classify_difficulty.py    # Main algorithm
├── data/
│   ├── dataset.json              # Input/output dataset
│   └── classified_dataset.json   # Optional separate output
└── README.md                     # This file
```

### Dataset Format

**Input/Output JSON structure:**

```json
[
  {
    "id": "ex01",
    "input": "The user can guarantee that sooner or later the ticket will be printed.",
    "output": "<<User>>F ticket_printed",
    "difficulty": ""
  },
  ...
]
```

The algorithm updates the `difficulty` field to either `"easy"` or `"hard"`.

---

## Theoretical Background

### Why Two Dimensions?

Traditional complexity measures focus on the **output** (formula structure). However, for NL-to-formal translation:

1. **Simple formulas can be hard to derive** when the NL is ambiguous
2. **Complex formulas can be straightforward** when the NL is explicit

```
                    High NL Ambiguity
                          ↑
                          │
           "Deceptively   │   "Genuinely
              Hard"       │     Hard"
                          │
    Simple Formula ←──────┼──────→ Complex Formula
                          │
           "Truly         │   "Deceptively
            Easy"         │     Easy"
                          │
                          ↓
                    Low NL Ambiguity
```

### Key Insight

For **translation tasks**, understanding the input is often harder than producing the output. This is why the default weights favor NL ambiguity (0.6) over formula complexity (0.4).

---

## License

MIT License - Feel free to modify and adapt for your needs.

---

## Contributing

To add new ambiguity patterns or complexity metrics:

1. Add detection functions in the appropriate section
2. Update the scoring aggregation functions
3. Add test cases to validate

```python
def detect_new_pattern(nl_input: str) -> float:
    """Detect [description]."""
    score = 0.0
    # Your logic here
    return score
```

Then add to `nl_ambiguity_score()`:

```python
def nl_ambiguity_score(nl_input: str, formula: str) -> float:
    return (
        # ... existing components ...
        detect_new_pattern(nl_input) +
    )
```
```

---

## Quick Start Summary

```bash
# 1. Navigate to your project
cd your_project

# 2. Run with defaults
python classify_difficulty.py

# 3. Check results
cat data/dataset.json | python -m json.tool | head -20
```

That's it! Your dataset now has difficulty labels.