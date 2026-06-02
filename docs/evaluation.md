# Evaluation

Evaluation is a two-step correctness check followed by optional judge agreement and reporting.

## Prediction Files

Experiments write JSON files under `outputs/model_predictions/`. Each row includes:

- `input`
- `expected`
- `expected_options`
- `generated`
- `raw_generation`
- `generation_prompt_sha256`
- `generation_config`
- `few_shot_example_ids` when few-shot prompting is enabled
- `token_usage` and `usage_estimated` when usage data is available
- `exact_match`
- optional latency fields

Top-level metadata records the dataset path/hash, model and experiment config paths/hashes, `pyproject.toml` and `uv.lock` hashes when present, command arguments, run timing, latency summaries, and a `split_manifest_path`/`split_manifest_sha256`. The split manifest under `outputs/split_manifests/` records ordered train, validation, and test membership by stable IDs and content hashes. Augmentation is deterministic from the training split, seed, and config, so augmented rows are not duplicated there.

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

Judged outputs are written to `outputs/LLM-evaluation/evaluated_datasets/<judge>/`. Each judged row keeps the parsed decision plus the prompt version, prompt hash, raw judge response, parse status, judge latency, and a decision method such as `exact`, `llm`, `no_llm`, or `unmatched`.

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

This builds judge summaries, agreement reports, seed aggregates, an accuracy-latency report, generated notebooks, and a `reproducibility_manifest.json` under `outputs/LLM-evaluation/`.

Seed aggregates are grouped by judge by default, so results from different judges are not silently pooled. Use `nl2atl aggregate-seeds --combine_judges` only when you intentionally want a combined exploratory view.

The manifest records input/report hashes, the current git commit when available, Python/platform details, and reproducibility limitations. Azure judge calls request `temperature=0`, but strict reproducibility still depends on preserving the Azure deployment mapping and any provider-side model snapshot guarantees. For publication claims, keep the raw predictions, split manifests, judged outputs, configs, lockfile, prompt versions, judge agreement report, generated notebooks, and any human-validation sample used to calibrate the judges.

## Accuracy-Latency Report

If you already have an aggregate file, build only the accuracy-latency report:

```bash
uv run nl2atl model-efficiency
```

The report keeps accuracy and latency separate, then adds deterministic helper rankings such as fastest model, most accurate model, best accuracy per second, highest throughput, and a Pareto frontier.
