# Evaluation

Evaluation is a two-step correctness check followed by optional judge agreement and reporting.

## Prediction Files

Experiments write JSON files under `outputs/model_predictions/`. Each row includes:

- `input`
- `expected`
- `expected_options`
- `generated`
- `exact_match`
- optional latency and difficulty fields

## Step 1: Exact Match

Exact match runs first. It normalizes formulas by lowercasing, removing whitespace, and normalizing common logical-symbol variants to the project's ASCII form.

Before comparison, model text is cleaned only at chat boundaries: prompt echoes and explicit assistant stop tokens are removed when present. The evaluator does not parse, repair, validate, or extract a best-looking ATL formula from a longer answer; malformed or verbose model output is kept in `generated` and evaluated as-is.

For rows with multiple gold formulas, exact match succeeds if the prediction matches any formula in `expected_options`.

## Step 2: LLM Judge

The LLM judge runs only for predictions that are not exact matches.

```bash
uv run nl2atl llm-judge --datasets all
```

By default, judge runs use GPT-5.2 and DeepSeek V3.2. GPT-4.1 and GPT-5.4 are reserved for generation baselines, so the strongest generator is not also the only evaluator in the main comparison.

The judge sees the natural-language input, the model prediction, and all accepted gold formulas. It returns:

```json
{ "correct": "yes", "reasoning": "..." }
```

Judged outputs are written to `outputs/LLM-evaluation/evaluated_datasets/<judge>/`.

## Agreement

Use agreement metrics when multiple judges evaluate the same predictions:

```bash
uv run nl2atl judge-agreement
```

The report includes pairwise Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, and optional disagreement examples.

## Combined Reports

```bash
uv run nl2atl generate-eval-reports
```

This builds judge summaries, agreement reports, seed aggregates, and an accuracy-latency report under `outputs/LLM-evaluation/`.

Seed aggregates are grouped by judge by default, so results from different judges are not silently pooled. Use `nl2atl aggregate-seeds --combine_judges` only when you intentionally want a combined exploratory view.

## Accuracy-Latency Report

If you already have an aggregate file, build only the accuracy-latency report:

```bash
uv run nl2atl model-efficiency
```

The report keeps accuracy and latency separate, then adds deterministic helper rankings such as fastest model, most accurate model, best accuracy per second, highest throughput, and a Pareto frontier.
