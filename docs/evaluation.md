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
- `latency_ms`

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

Useful options:

| Option | Purpose |
|---|---|
| `--datasets all` | Evaluate all prediction JSON files under `outputs/model_predictions/` |
| `--datasets file.json` | Evaluate one prediction file by path, name, or name without `.json` |
| `--models gpt-5.2` | Select one or more Azure judge models; aliases include `--model`, `--judge_model`, and `--judge_models` |
| `--no_llm` | Write judged artifacts using exact-match decisions only, without API calls |
| `--overwrite` | Re-evaluate existing judged files |
| `--predictions_dir` / `--output_dir` | Override input and output directories |

The judge sees the natural-language input, the model prediction, and all accepted gold formulas. It returns:

```json
{ "correct": "yes", "reasoning": "..." }
```

Judged outputs are written to `outputs/LLM-evaluation/evaluated_datasets/<judge>/`. Each judged row keeps the parsed decision plus the prompt version, prompt hash, raw judge response, parse status, judge latency, and a decision method such as `exact`, `llm`, `no_llm`, or `unmatched`.

Existing judged files are reused when their prompt version matches the current judge prompt. If the prompt version changes, they are regenerated.

## Agreement

Use agreement metrics when multiple judges evaluate the same predictions:

```bash
uv run nl2atl judge-agreement
```

The report includes pairwise Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, and optional disagreement examples.

## Human Judge Calibration Sample

For publication-facing human validation, create a blind stratified audit set from the paired LLM-judge outputs:

```bash
uv run nl2atl human-eval-sample
```

The default package is written to `outputs/LLM-evaluation/human_evaluation/`. It includes a 600-item AAAI-oriented core sample, a master blind XLSX workbook, two annotator-specific XLSX workbooks, a keyed metadata file for post-annotation analysis, and a protocol document. Exact matches are excluded from the default human workload because they are accepted by deterministic normalization before LLM judging. The XLSX files restrict `correct` to `yes`/`no` and `annotator_id` to `Francesco`/`Marco`. The blind files intentionally hide generator identity, judge identity, judge decisions, judge reasoning, and sampling strata from annotators. CSV/JSON/JSONL blind files and the full disagreement pool can be generated with `--legacy_formats` and `--write_disagreement_pool` when needed, but they are not required for the normal annotation workflow.

The default core sample is enriched for judge calibration rather than raw population prevalence: all unique rare reverse disagreements are included, common disagreements are oversampled, and LLM agreement controls are retained. The sample is designed to answer whether the LLM judges are trustworthy on cases where deterministic exact match cannot decide correctness.

Sampling starts from the full paired-judge population of 13,200 evaluated prediction items, each judged by DeepSeek V3.2 and GPT-5.2. Exact matches are removed because they are accepted automatically before LLM judging. The remaining items are grouped by judge-decision stratum, with disagreement strata oversampled because they reveal which judge is closer to human expert labels. Agreement controls are retained to test whether consensus labels are reliable: `llm_agree_yes` detects overly permissive consensus, and `llm_agree_no` detects overly strict consensus. Duplicate `input` + `gold_options` + `prediction` triples are collapsed in the core sample to avoid redundant annotation, while the private key file keeps the hidden source metadata. Within each stratum, examples are spread across generator model, condition, and seed so the sample is not dominated by one run.

Current default sample composition:

| Stratum | Meaning | Count |
|---|---|---:|
| `disagree_ds_no_gpt_yes` | DeepSeek rejects, GPT accepts | 26 |
| `disagree_ds_yes_gpt_no` | DeepSeek accepts, GPT rejects | 334 |
| `llm_agree_no` | both judges reject | 120 |
| `llm_agree_yes` | both judges accept | 120 |
| **Total** |  | **600** |

Use two ATL-literate project annotators when possible. They should annotate independently first, then resolve disagreements through a documented adjudication/discussion pass. After annotation, report human-human agreement before adjudication, the number of human-human disagreements, LLM-human agreement per judge, adjudicated accuracy by stratum, the deterministic exact-match policy, and whether the main model ranking is stable under human adjudication.

After annotators complete their blind XLSX workbooks, merge them with the private key:

```bash
uv run nl2atl human-eval-merge outputs/LLM-evaluation/human_evaluation/annotations/Francesco_blind.xlsx outputs/LLM-evaluation/human_evaluation/annotations/Marco_blind.xlsx
```

Annotators only need to fill the `correct` column with `yes` or `no`. Use the XLSX templates to avoid typos in the `correct` and `annotator_id` columns. The merge command reads completed XLSX files and writes analysis-ready CSV, JSON, and JSONL files under `outputs/LLM-evaluation/human_evaluation/merged/`, with human labels, human-human status, per-judge LLM-human agreement fields, generator metadata, judge decisions, and hidden sampling strata joined by `audit_id`. Human-human disagreements are marked with `needs_adjudication=yes`, `human_final_correct=pending_adjudication`, and per-annotator match columns, so they remain analyzable before the final discussion pass. Free-text human reasoning is intentionally not part of the annotation or merged output; keep any adjudication notes separately only when a disagreement needs discussion.

To regenerate only the blank Francesco/Marco annotation workbooks from the existing 600-item key, without resampling the audit set, run:

```bash
uv run nl2atl human-eval-sample --regenerate_annotator_workbooks
```

The command writes new files under `outputs/LLM-evaluation/human_evaluation/annotations/` and backs up existing annotator workbooks before replacing them.

## Combined Reports

```bash
uv run nl2atl generate-eval-reports
```

This builds judge summaries, agreement reports, seed aggregates, an accuracy-latency report, one publication-focused notebook at `outputs/LLM-evaluation/publication_analysis.ipynb`, and a `reproducibility_manifest.json`.

The pipeline has skip flags for each stage: `--skip_judge_summaries`, `--skip_agreement`, `--skip_seed_aggregation`, and `--skip_efficiency`. Use `--no_notebook` to skip the unified notebook, `--notebook_output` to change its path, or `--individual_notebooks` when you also need the older per-report notebooks.

The publication notebook is intentionally compact for paper writing: final judged accuracy, seed variability, exact-match versus LLM-judge contribution, judge reliability, accuracy-latency Pareto analysis, and a reproducibility snapshot. Use `--individual_notebooks` only when you explicitly need the older per-report notebooks for debugging.

Seed aggregates are grouped by judge by default, so results from different judges are not silently pooled. Use `nl2atl aggregate-seeds --combine_judges` only when you intentionally want a combined exploratory view.

The manifest records input/report hashes, the current git commit when available, Python/platform details, and reproducibility limitations. Azure judge calls request `temperature=0`, but strict reproducibility still depends on preserving the Azure deployment mapping and any provider-side model snapshot guarantees. For publication claims, keep the raw predictions, split manifests, judged outputs, configs, lockfile, prompt versions, judge agreement report, publication notebook, and any human-validation sample used to calibrate the judges.

## Accuracy-Latency Report

If you already have an aggregate file, build only the accuracy-latency report:

```bash
uv run nl2atl model-efficiency
```

The report keeps accuracy and latency separate, then adds deterministic helper rankings such as fastest model, most accurate model, best accuracy per second, highest throughput, and a Pareto frontier.

Options include `--top_k`, `--include_per_seed`, `--weight_accuracy`, `--weight_latency`, and `--no_notebook`. The report intentionally avoids monetary pricing, GPU-hour estimates, and token-derived statistics; use it for accuracy and latency comparisons only.
