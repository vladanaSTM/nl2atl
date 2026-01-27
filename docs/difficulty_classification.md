# Difficulty classification for NL → ATL

This document describes the current difficulty classifier used by NL2ATL to label dataset examples as
`easy` or `hard`. The implementation lives in [src/evaluation/difficulty.py](../src/evaluation/difficulty.py)
and is exposed via the `nl2atl` CLI.

## Overview

The classifier combines two signals:

1) **Formula complexity** — syntactic complexity of the ATL output.
2) **NL ambiguity** — linguistic ambiguity in the natural-language input.

Each example is assigned a combined score and labeled `hard` if the score is above a threshold.

## CLI usage

From the repo root:

```bash
nl2atl classify-difficulty --input data/dataset.json --verbose
```

You can also run the module directly:

```bash
python -m src.cli.classify_difficulty --input data/dataset.json --verbose
```

Key options (defaults shown):

- `--input/-i`: input JSON (default: `data/dataset.json`)
- `--output/-o`: output JSON (default: same as input)
- `--formula-weight/-fw`: 0.4
- `--nl-weight/-nw`: 0.6
- `--threshold/-t`: 5.0
- `--verbose/-v`: print per‑example details

## Output format

Each dataset item is updated in place with:

- `difficulty`: `easy` or `hard`
- `difficulty_scores`: numeric breakdown

Example output fields:

```json
{
  "difficulty": "hard",
  "difficulty_scores": {
    "formula_complexity": 5.5,
    "nl_ambiguity": 4.0,
    "combined_score": 4.6,
    "threshold": 5.0
  }
}
```

## Scoring model

The combined score is:

$$
	ext{combined} = w_f \cdot \text{formula\_complexity} + w_{nl} \cdot \text{nl\_ambiguity}
$$

Default weights: $w_f = 0.4$, $w_{nl} = 0.6$.

Label rule:

- `hard` if $\text{combined} > \text{threshold}$
- `easy` otherwise

### Formula complexity

Components and weights (current implementation):

- Temporal operators $G,F,X,U,W,R$: $1.0$ each
- Nesting depth: $1.5$ per depth level (with a small bonus when multiple depths occur)
- Logical connectives: $0.5$ each (`->`, `→`, `&&`, `∧`, `||`, `∨`, `!`, `¬`)
- Coalition size: $0.5$ for each additional agent beyond 1 in `<<A,B,...>>`

### NL ambiguity

The score sums several detectors:

- **Scope ambiguity**: patterns like “if … at the next step” or “always … if … then”
- **Implicit operators**: temporal operators present in the formula but not stated in NL
  - `G`: missing “always/forever/at all times” cues
  - `F`: missing “eventually/sooner or later” cues
  - `X`: missing “next step/next state” cues
  - `U`: missing “until” cues
- **Semantic gap**: phrases like “leave”, “avoid”, “prevent”, and negation mismatch
- **Pronoun complexity**: pronoun frequency (`they`, `it`, `their`, …)
- **Temporal ambiguity**: “after”, “before”, “while”, “unless”, …
- **Agent complexity**: large coalitions and “together/jointly/both … and” patterns
- **Structural complexity**: multiple conditionals, clause markers, and sentence length

## Tuning guidelines

- Increase `--threshold` to make `hard` less frequent.
- Increase `--nl-weight` if you want ambiguity to dominate.
- Increase `--formula-weight` if you want formula structure to dominate.

For typical datasets, start with defaults and adjust using the verbose output summary.

## Programmatic API

```python
from src.evaluation.difficulty import (
    classify_difficulty,
    process_dataset,
    formula_complexity_score,
    nl_ambiguity_score,
)

label, scores = classify_difficulty(
    nl_input="The user can guarantee that after payment, a ticket is eventually printed.",
    formula="<<User>>G (paid -> F ticket_printed)",
)

dataset = process_dataset(
    input_path="data/dataset.json",
    output_path="data/dataset_with_difficulty.json",
    verbose=True,
)
```

## Related docs

- Usage overview: [usage.md](usage.md)
- Dataset format: [dataset.md](dataset.md)