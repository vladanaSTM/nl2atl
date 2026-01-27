# Usage Guide

This comprehensive guide covers all NL2ATL command-line workflows, from running experiments to analyzing results. Each command includes detailed examples, options, and troubleshooting tips.

## Table of Contents

1. [CLI Overview](#cli-overview)
2. [Running Experiments](#running-experiments)
3. [Evaluation Commands](#evaluation-commands)
4. [Analysis and Reporting](#analysis-and-reporting)
5. [Utility Commands](#utility-commands)
6. [Common Workflows](#common-workflows)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## CLI Overview

NL2ATL provides a unified command-line interface through the `nl2atl` command:

```bash
nl2atl --help
```

### Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `run-single` | Run one experiment | Testing, debugging, or single model evaluation |
| `run-all` | Run experiment sweep | Local multi-model/condition testing |
| `run-array` | Run SLURM array task | Parallel execution on HPC clusters |
| `llm-judge` | LLM-based evaluation | Semantic correctness checking |
| `judge-agreement` | Inter-rater agreement | Comparing multiple judges or humans |
| `model-efficiency` | Cost-latency-accuracy analysis | Model selection and comparison |
| `classify-difficulty` | Dataset difficulty labeling | Understanding dataset complexity |

### Global Options

```bash
nl2atl COMMAND --help  # Show help for specific command
```

---

## Running Experiments

### Single Experiment: `run-single`

Run a single experiment with one model and configuration.

#### Basic Usage

```bash
nl2atl run-single --model MODEL_KEY [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | string | required | Model key from `models.yaml` |
| `--finetuned` | flag | false | Use fine-tuned adapter |
| `--few_shot` | flag | false | Enable few-shot prompting |
| `--num_few_shot` | int | 5 | Number of few-shot examples |
| `--seed` | int | 42 | Random seed |
| `--adapter` | string | - | Specific adapter path |
| `--max_new_tokens` | int | 128 | Max generation length |
| `--batch_size` | int | 8 | Inference batch size |
| `--overwrite` | flag | false | Overwrite existing output |

#### Examples

**Zero-shot baseline:**
```bash
nl2atl run-single --model qwen-3b
```

**Few-shot prompting:**
```bash
nl2atl run-single --model qwen-3b --few_shot
```

**Custom few-shot count:**
```bash
nl2atl run-single --model qwen-3b --few_shot --num_few_shot 10
```

**Fine-tuned model:**
```bash
nl2atl run-single --model qwen-3b --finetuned
```

**Fine-tuned with few-shot:**
```bash
nl2atl run-single --model qwen-3b --finetuned --few_shot
```

**Custom adapter path:**
```bash
nl2atl run-single --model qwen-3b --adapter models/qwen-3b_custom/final
```

**Specific seed for reproducibility:**
```bash
nl2atl run-single --model qwen-3b --few_shot --seed 123
```

**Force re-run existing experiment:**
```bash
nl2atl run-single --model qwen-3b --few_shot --overwrite
```

#### Output

Predictions saved to:
```
outputs/model_predictions/<model>_<condition>_seed<seed>.json
```

Example output:
```
Loading model: Qwen/Qwen2.5-3B-Instruct
Preparing dataset (seed=42)...
  Train: 210 examples (augmented: 2100)
  Test: 90 examples

Running inference...
Progress: 90/90 [████████████████████] 100% 

Results:
  Exact match: 74/90 (82.2%)
  Mean latency: 520.1 ms
  P95 latency: 910.5 ms
  Total time: 2m 15s

Saved: outputs/model_predictions/qwen-3b_baseline_few_shot_seed42.json
```

---

### Experiment Sweep: `run-all`

Run multiple experiments with various model/condition combinations.

#### Basic Usage

```bash
nl2atl run-all [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--models` | list | all | Model keys to test (space-separated) |
| `--conditions` | list | all | Condition names (space-separated) |
| `--seeds` | list | [42] | Seeds to use |
| `--model_provider` | string | all | Filter by provider (`hf`, `azure`, `all`) |
| `--overwrite` | flag | false | Overwrite existing outputs |
| `--force` | flag | false | Alias for `--overwrite` |
| `--parallel` | int | 1 | Number of parallel workers |

#### Examples

**Run all models and conditions:**
```bash
nl2atl run-all
```

**Specific model, all conditions:**
```bash
nl2atl run-all --models qwen-3b
```

**Specific models, specific conditions:**
```bash
nl2atl run-all --models qwen-3b llama-8b --conditions baseline_zero_shot baseline_few_shot
```

**Multiple seeds:**
```bash
nl2atl run-all --models qwen-3b --seeds 42 43 44 45 46
```

**Only HuggingFace models:**
```bash
nl2atl run-all --model_provider hf
```

**Only Azure models:**
```bash
nl2atl run-all --model_provider azure
```

**Re-run everything:**
```bash
nl2atl run-all --overwrite
```

**Parallel execution (local):**
```bash
nl2atl run-all --models qwen-3b llama-8b --parallel 2
```

#### Output

Multiple prediction files:
```
outputs/model_predictions/
├── qwen-3b_baseline_zero_shot_seed42.json
├── qwen-3b_baseline_few_shot_seed42.json
├── llama-8b_baseline_zero_shot_seed42.json
└── llama-8b_baseline_few_shot_seed42.json
```

---

### SLURM Array Jobs: `run-array`

Run experiments as SLURM array tasks for parallel execution on HPC clusters.

#### Task Inspection

**Count total tasks:**
```bash
nl2atl run-array --count
```

**List all tasks:**
```bash
nl2atl run-array --list-tasks
```

**Filter and count:**
```bash
nl2atl run-array --models qwen-3b --conditions baseline_zero_shot --count
```

**Filter by seed:**
```bash
nl2atl run-array --seed 123 --count
```

#### Task Execution

**Run specific array task:**
```bash
nl2atl run-array --task-id 0
```

This is typically called by SLURM, not manually.

#### SLURM Submission

Use the provided submission script:

```bash
sbatch scripts/slurm/submit_array.sh
```

**Edit `submit_array.sh` to customize:**
- Resources (GPUs, memory, time)
- Model/condition filters
- Output directory

Example `submit_array.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=nl2atl
#SBATCH --output=logs/nl2atl_%A_%a.out
#SBATCH --error=logs/nl2atl_%A_%a.err
#SBATCH --array=0-19%4          # 20 tasks, max 4 concurrent
#SBATCH --gres=gpu:1            # 1 GPU per task
#SBATCH --mem=16G               # 16 GB RAM
#SBATCH --time=02:00:00         # 2 hour time limit

# Activate environment
source .venv/bin/activate

# Run array task
nl2atl run-array --task-id $SLURM_ARRAY_TASK_ID
```

#### Monitoring SLURM Jobs

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel JOB_ID

# Check output logs
tail -f logs/nl2atl_JOBID_0.out
```

**See [SLURM Guide](slurm.md) for comprehensive SLURM documentation.**

---

## Evaluation Commands

### LLM Judge: `llm-judge`

Evaluate predictions using LLM-as-judge for semantic correctness.

#### Basic Usage

```bash
nl2atl llm-judge --datasets DATASETS [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--datasets` | list | required | Prediction files or `all` |
| `--judge_model` | string | gpt-5.2 | Judge model key |
| `--judge_models` | list | - | Multiple judges (overrides `--judge_model`) |
| `--predictions_dir` | path | outputs/model_predictions | Predictions directory |
| `--output_dir` | path | outputs/LLM-evaluation | Output directory |
| `--overwrite` | flag | false | Re-evaluate existing files |
| `--force` | flag | false | Alias for `--overwrite` |
| `--no_llm` | flag | false | Skip LLM, use exact match only |
| `--batch_size` | int | 10 | Batch size for LLM calls |

#### Examples

**Evaluate all predictions:**
```bash
nl2atl llm-judge --datasets all
```

**Evaluate specific file:**
```bash
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot_seed42.json
```

**Evaluate multiple files:**
```bash
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot.json llama-8b_baseline_zero_shot.json
```

**Use different judge:**
```bash
nl2atl llm-judge --datasets all --judge_model gpt-4
```

**Multiple judges:**
```bash
nl2atl llm-judge --datasets all --judge_models gpt-4 gpt-5.2 claude-3.5
```

**Re-evaluate existing:**
```bash
nl2atl llm-judge --datasets all --overwrite
```

**Exact match only (fast):**
```bash
nl2atl llm-judge --datasets all --no_llm
```

#### Output

Evaluated files:
```
outputs/LLM-evaluation/
├── evaluated_datasets/
│   ├── gpt-5.2/
│   │   └── qwen-3b_baseline_few_shot__judge-gpt-5.2.json
│   └── gpt-4/
│       └── qwen-3b_baseline_few_shot__judge-gpt-4.json
└── summary__judge-gpt-5.2.json
```

#### Output Format

Each evaluated file contains:
```json
{
  "run_id": "qwen-3b_baseline_few_shot_seed42",
  "judge_model": "gpt-5.2",
  "metrics": {
    "n_examples": 90,
    "exact_match": 0.822,
    "llm_evaluated": 16,
    "llm_correct": 8,
    "total_correct": 82,
    "accuracy": 0.911
  },
  "detailed_results": [
    {
      "input": "...",
      "gold": "<<User>>F p",
      "prediction": "<<User>>F p",
      "correct": "yes",
      "reasoning": "Exact match (normalized)",
      "decision_method": "exact"
    }
  ]
}
```

---

### Judge Agreement: `judge-agreement`

Compute inter-rater agreement metrics across multiple judges.

#### Basic Usage

```bash
nl2atl judge-agreement --eval_dir EVAL_DIR [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--eval_dir` | path | required | Directory with evaluated datasets |
| `--output_path` | path | outputs/LLM-evaluation/agreement_report.json | Output file |
| `--judges` | list | all | Specific judge folders to include |
| `--human_annotations` | path | - | Human annotations file |
| `--include_disagreements` | flag | true | Include disagreement examples |
| `--no_disagreements` | flag | false | Exclude disagreement examples |
| `--max_disagreements` | int | 20 | Max disagreements to include |

#### Examples

**Basic agreement analysis:**
```bash
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
```

**Specific judges:**
```bash
nl2atl judge-agreement \
  --eval_dir outputs/LLM-evaluation/evaluated_datasets \
  --judges gpt-4 gpt-5.2
```

**Include human annotations:**
```bash
nl2atl judge-agreement \
  --eval_dir outputs/LLM-evaluation/evaluated_datasets \
  --human_annotations data/human_annotations.json
```

**No disagreement examples:**
```bash
nl2atl judge-agreement \
  --eval_dir outputs/LLM-evaluation/evaluated_datasets \
  --no_disagreements
```

**Limit disagreement examples:**
```bash
nl2atl judge-agreement \
  --eval_dir outputs/LLM-evaluation/evaluated_datasets \
  --max_disagreements 5
```

#### Output

Agreement report with:
- **Pairwise Cohen's κ** between each judge pair
- **Fleiss' κ** for all judges (if same samples)
- **Krippendorff's α** (handles missing data)
- **Disagreement examples** (optional)

Example output:
```json
{
  "judges": ["gpt-4", "gpt-5.2", "human"],
  "total_items": 90,
  "pairwise_agreements": {
    "gpt-4 vs gpt-5.2": {
      "cohens_kappa": 0.856,
      "agreement_rate": 0.911,
      "n_agreements": 82,
      "n_disagreements": 8
    },
    "gpt-4 vs human": {
      "cohens_kappa": 0.792,
      "agreement_rate": 0.878
    }
  },
  "fleiss_kappa": 0.824,
  "krippendorff_alpha": 0.831,
  "disagreements": [
    {
      "id": "ex42",
      "input": "...",
      "gold": "...",
      "prediction": "...",
      "judges": {
        "gpt-4": "yes",
        "gpt-5.2": "no",
        "human": "yes"
      }
    }
  ]
}
```

---

## Analysis and Reporting

### Model Efficiency: `model-efficiency`

Generate comprehensive efficiency report with cost-latency-accuracy trade-offs.

#### Basic Usage

```bash
nl2atl model-efficiency --predictions_dir PRED_DIR [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--predictions_dir` | path | required | Predictions directory |
| `--output_path` | path | outputs/LLM-evaluation/efficiency_report.json | Output file |
| `--judge_summary` | path | - | Judge summary file (for LLM accuracy) |
| `--judge_model` | string | - | Auto-find judge summary by model |
| `--weight_accuracy` | float | 0.4 | Accuracy weight in composite score |
| `--weight_cost` | float | 0.3 | Cost weight in composite score |
| `--weight_latency` | float | 0.3 | Latency weight in composite score |
| `--top_k` | int | 10 | Size of ranking lists |
| `--no_notebook` | flag | false | Skip notebook generation |

#### Examples

**Basic efficiency report:**
```bash
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

**Use LLM-judge accuracy:**
```bash
nl2atl model-efficiency \
  --predictions_dir outputs/model_predictions \
  --judge_model gpt-5.2
```

**Custom weights (prioritize accuracy):**
```bash
nl2atl model-efficiency \
  --predictions_dir outputs/model_predictions \
  --weight_accuracy 0.6 \
  --weight_cost 0.2 \
  --weight_latency 0.2
```

**Custom weights (prioritize cost):**
```bash
nl2atl model-efficiency \
  --predictions_dir outputs/model_predictions \
  --weight_accuracy 0.3 \
  --weight_cost 0.5 \
  --weight_latency 0.2
```

**Skip notebook (faster):**
```bash
nl2atl model-efficiency \
  --predictions_dir outputs/model_predictions \
  --no_notebook
```

#### Output Files

1. **JSON Report**: `efficiency_report.json`
2. **Jupyter Notebook**: `efficiency_report.ipynb` (unless `--no_notebook`)

#### Report Structure

```json
{
  "overall_stats": {
    "total_runs": 8,
    "total_formulas": 720,
    "models_tested": ["qwen-3b", "llama-8b"],
    "conditions_tested": ["baseline_zero_shot", "baseline_few_shot"]
  },
  "rankings": {
    "by_accuracy": [
      {
        "run_id": "qwen-3b_baseline_few_shot",
        "accuracy": 0.911,
        "rank": 1
      }
    ],
    "by_cost": [
      {
        "run_id": "qwen-3b_baseline_zero_shot",
        "total_cost_usd": 0.45,
        "cost_per_formula_usd": 0.005,
        "rank": 1
      }
    ],
    "by_latency": [
      {
        "run_id": "qwen-3b_baseline_zero_shot",
        "latency_mean_ms": 380.2,
        "throughput": 2.63,
        "rank": 1
      }
    ],
    "by_composite": [
      {
        "run_id": "qwen-3b_baseline_few_shot",
        "composite_score": 0.847,
        "normalized_accuracy": 0.95,
        "normalized_cost": 0.78,
        "normalized_latency": 0.82,
        "rank": 1
      }
    ]
  }
}
```

---

## Utility Commands

### Classify Difficulty: `classify-difficulty`

Label dataset samples by difficulty using rule-based classifier.

#### Basic Usage

```bash
nl2atl classify-difficulty --input INPUT_FILE [OPTIONS]
```

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input`, `-i` | path | data/dataset.json | Input dataset |
| `--output`, `-o` | path | same as input | Output dataset |
| `--formula-weight`, `-fw` | float | 0.4 | Formula complexity weight |
| `--nl-weight`, `-nw` | float | 0.6 | NL ambiguity weight |
| `--threshold`, `-t` | float | 5.0 | Easy/hard threshold |
| `--verbose`, `-v` | flag | false | Print per-sample details |

#### Examples

**Basic classification:**
```bash
nl2atl classify-difficulty --input data/dataset.json
```

**With detailed output:**
```bash
nl2atl classify-difficulty --input data/dataset.json --verbose
```

**Save to different file:**
```bash
nl2atl classify-difficulty \
  --input data/dataset.json \
  --output data/dataset_labeled.json
```

**Custom weights:**
```bash
nl2atl classify-difficulty \
  --input data/dataset.json \
  --formula-weight 0.5 \
  --nl-weight 0.5 \
  --threshold 4.5
```

**Prioritize NL ambiguity:**
```bash
nl2atl classify-difficulty \
  --input data/dataset.json \
  --formula-weight 0.3 \
  --nl-weight 0.7
```

#### Output

Dataset with added `difficulty` and `difficulty_scores` fields:

```json
{
  "id": "ex42",
  "input": "...",
  "output": "<<A>>G (p -> F q)",
  "difficulty": "hard",
  "difficulty_scores": {
    "formula_complexity": 6.5,
    "nl_ambiguity": 4.2,
    "combined_score": 5.1,
    "threshold": 5.0
  }
}
```

**See [Difficulty Classification](difficulty_classification.md) for scoring methodology.**

---

## Common Workflows

### Complete Experiment Pipeline

```bash
# 1. Run experiments
nl2atl run-all --models qwen-3b llama-8b

# 2. Evaluate with LLM judge
nl2atl llm-judge --datasets all

# 3. Compute agreement
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets

# 4. Generate efficiency report
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

### Quick Single Model Test

```bash
# Run and evaluate in sequence
nl2atl run-single --model qwen-3b --few_shot
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot_seed42.json
```

### Multi-Seed Reproducibility Study

```bash
# Run same config with different seeds
nl2atl run-all --models qwen-3b --conditions baseline_few_shot --seeds 42 43 44 45 46

# Evaluate all
nl2atl llm-judge --datasets all

# Analyze variance
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

### Model Comparison

```bash
# Run baseline and fine-tuned
nl2atl run-all --models qwen-3b --conditions baseline_few_shot finetuned_few_shot

# Evaluate
nl2atl llm-judge --datasets all

# Compare efficiency
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

---

## Output Files

### Directory Structure

```
outputs/
├── model_predictions/
│   ├── qwen-3b_baseline_zero_shot_seed42.json
│   ├── qwen-3b_baseline_few_shot_seed42.json
│   └── ...
└── LLM-evaluation/
    ├── evaluated_datasets/
    │   ├── gpt-5.2/
    │   │   ├── qwen-3b_baseline_zero_shot__judge-gpt-5.2.json
    │   │   └── ...
    │   └── gpt-4/
    │       └── ...
    ├── summary__judge-gpt-5.2.json
    ├── summary__judge-gpt-4.json
    ├── efficiency_report.json
    ├── efficiency_report.ipynb
    └── agreement_report.json
```

### Prediction File Format

See [Output Files](quickstart.md#step-4-inspect-predictions) in Quick Start for detailed format.

---

## Troubleshooting

### Common Issues

#### "Model not found in config"

**Problem**: Model key not in `models.yaml`

**Solution**:
```bash
# List available models
cat configs/models.yaml | grep "^  [a-z]"

# Use exact key name
nl2atl run-single --model qwen-3b  # not "qwen3b" or "Qwen-3B"
```

#### "CUDA out of memory"

**Problem**: GPU memory exhausted

**Solutions**:
1. Enable 4-bit quantization in `models.yaml`:
   ```yaml
   load_in_4bit: true
   ```
2. Reduce batch size:
   ```bash
   nl2atl run-single --model qwen-3b --batch_size 4
   ```
3. Use smaller model or Azure API

#### "Predictions file already exists"

**Problem**: Output file exists, won't overwrite

**Solution**:
```bash
# Force overwrite
nl2atl run-single --model qwen-3b --overwrite

# Or delete manually
rm outputs/model_predictions/qwen-3b_baseline_zero_shot_seed42.json
```

#### "Azure authentication failed"

**Problem**: Invalid Azure credentials

**Solution**:
1. Check `.env` file:
   ```bash
   cat .env | grep AZURE
   ```
2. Verify credentials are correct
3. Test Azure endpoint:
   ```bash
   curl -H "api-key: $AZURE_API_KEY" "$AZURE_INFER_ENDPOINT/health"
   ```

#### "No predictions found for judge"

**Problem**: Judge can't find prediction files

**Solution**:
```bash
# List available predictions
ls outputs/model_predictions/

# Use exact filename
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot_seed42.json
```

#### "Task ID out of range"

**Problem**: SLURM array index exceeds task count

**Solution**:
```bash
# Check total tasks
nl2atl run-array --count

# Ensure SLURM array size matches
#SBATCH --array=0-19  # If count is 20
```

### Getting Help

```bash
# Command-specific help
nl2atl run-single --help
nl2atl llm-judge --help

# General help
nl2atl --help
```

---

## Advanced Options

### Environment Variables

Override defaults with environment variables:

```bash
# Override default model
NL2ATL_DEFAULT_MODEL=llama-8b nl2atl run-single

# Custom config paths
NL2ATL_MODELS_CONFIG=/path/to/models.yaml \
NL2ATL_EXPERIMENTS_CONFIG=/path/to/experiments.yaml \
nl2atl run-all
```

### Python Module Invocation

Alternative to `nl2atl` command:

```bash
# Run single experiment
python -m src.cli.run_single_experiment --model qwen-3b --few_shot

# Run all experiments
python -m src.cli.run_all_experiments --models qwen-3b

# LLM judge
python -m src.cli.llm_judge --datasets all
```

---

## Next Steps

- **Configuration**: See [Configuration Guide](configuration.md) for config file details
- **Evaluation**: See [Evaluation Guide](evaluation.md) for metric explanations
- **SLURM**: See [SLURM Guide](slurm.md) for HPC cluster usage
- **API**: See [API Reference](api.md) for programmatic usage

---

**Questions?** Check the [documentation index](index.md) or [open an issue](https://github.com/vladanaSTM/nl2atl/issues).
