# NL2ATL: Natural Language to Alternating-Time Temporal Logic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NL2ATL** is a comprehensive research framework for translating natural language specifications into Alternating-Time Temporal Logic (ATL) formulas using Large Language Models. This project provides end-to-end infrastructure for experimentation, evaluation, and deployment of NL-to-ATL translation models.

## Overview

Multi-agent systems require formal specifications to verify properties like safety, liveness, and strategic guarantees. ATL is a powerful temporal logic for expressing what coalitions of agents can guarantee regardless of how other agents behave. However, writing ATL formulas requires expertise. NL2ATL bridges this gap by enabling users to express requirements in natural language and automatically generating correct ATL formulas.

### Key Features

- 🚀 **Comprehensive Experiment Framework**: Run baseline and fine-tuned models with stratified data splits, automated evaluation, and reproducible experiment management
- 🎯 **Multi-Model Support**: Compatible with Hugging Face models (Qwen, Llama, Mistral, etc.) and Azure OpenAI endpoints
- 📊 **Advanced Evaluation Pipeline**: Exact-match scoring, LLM-as-judge evaluation, inter-rater agreement metrics (Cohen's κ, Fleiss' κ, Krippendorff's α)
- 💰 **Efficiency Analysis**: Cost-latency-accuracy trade-off reporting with composite efficiency scores
- 🎓 **Difficulty Classification**: Rule-based classifier for labeling dataset complexity
- 🔌 **API Service**: FastAPI endpoint for UI integration (e.g., genVITAMIN model checker)
- ⚡ **SLURM Support**: Parallel experiment execution with scheduler-managed resources
- 📝 **Rich Documentation**: Comprehensive guides covering installation, configuration, usage, and extension

## Common Tasks

```bash
# Run a single experiment
nl2atl run-single --model qwen-3b --few_shot

# Run a sweep across models and conditions
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot

# Evaluate outputs with an LLM judge
nl2atl llm-judge --datasets all

# Generate efficiency report
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

---

## 📦 Accessing Large Output Files (Git LFS)

Some files in the `outputs/` directory (such as model predictions and evaluation results) are tracked using [Git Large File Storage (LFS)](https://git-lfs.github.com/).

**To clone and use these files:**

1. Install Git LFS:
  ```bash
  # On Ubuntu/Debian
  sudo apt-get install git-lfs
  # Or use Homebrew (macOS)
  brew install git-lfs
  ```
2. Initialize Git LFS (once per machine):
  ```bash
  git lfs install
  ```
3. Clone the repository as usual:
  ```bash
  git clone https://github.com/vladanaSTM/nl2atl.git
  cd nl2atl
  ```
4. Pull LFS files (if needed):
  ```bash
  git lfs pull
  ```

If you do not install Git LFS, you will only see small pointer files instead of the actual data in `outputs/`.

For more details, see: https://git-lfs.github.com/

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set required variables:
# - AZURE_API_KEY, AZURE_INFER_ENDPOINT (if using Azure models)
# - HUGGINGFACE_TOKEN (if using gated models)
# - WANDB_API_KEY (optional, for experiment tracking)
```

### Run Your First Experiment

```bash
# Single experiment with a 3B parameter model
nl2atl run-single --model qwen-3b --few_shot

# Full experiment sweep
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot

# Evaluate with LLM judge
nl2atl llm-judge --datasets all

# Generate efficiency report
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

## What is ATL?

Alternating-Time Temporal Logic (ATL) expresses strategic capabilities of agent coalitions in multi-agent systems. Unlike standard temporal logics that describe what *will* happen, ATL describes what agents *can force* to happen.

**Example translations:**

| Natural Language | ATL Formula |
|-----------------|-------------|
| "The user can guarantee the ticket is eventually printed" | `<<User>>F ticket_printed` |
| "The controller can always avoid errors" | `<<Controller>>G !error` |
| "Agents A and B together can keep the system safe until recovery" | `<<A,B>>(safe U recovered)` |

### ATL Syntax

- **Coalition modality**: `<<A,B>>` (agents A and B have a joint strategy)
- **Temporal operators**:
  - `X p` — next state
  - `F p` — eventually 
  - `G p` — always
  - `p U q` — until
- **Logical operators**: `!` (not), `&&` (and), `||` (or), `->` (implies)

See [docs/atl_primer.md](docs/atl_primer.md) for a complete introduction.

## Project Structure

```
nl2atl/
├── nl2atl.py                    # CLI entrypoint
├── src/
│   ├── cli/                     # Command-line interface modules
│   │   ├── main.py             # Main CLI dispatcher
│   │   ├── run_*.py            # Experiment runners
│   │   └── ...
│   ├── experiment/              # Experiment orchestration
│   │   ├── runner.py           # ExperimentRunner
│   │   ├── data_manager.py     # Data splitting and augmentation
│   │   └── reporter.py         # Result persistence
│   ├── models/                  # Model loading and inference
│   │   ├── registry.py         # Model registry
│   │   ├── model_*.py          # Provider-specific loaders
│   │   └── prompt_formatter.py # Prompt templates
│   ├── evaluation/              # Evaluation pipelines
│   │   ├── exact_match.py      # String-based evaluation
│   │   ├── llm_judge/          # LLM-as-judge evaluation
│   │   ├── agreement.py        # Inter-rater agreement
│   │   ├── efficiency.py       # Cost-latency-accuracy analysis
│   │   └── difficulty.py       # Difficulty classification
│   ├── infra/                   # Infrastructure utilities
│   │   ├── io.py               # File I/O helpers
│   │   └── azure_client.py     # Azure API client
│   └── api_server.py            # FastAPI service
├── configs/
│   ├── models.yaml             # Model registry configuration
│   └── experiments.yaml        # Experiment conditions
├── data/
│   └── dataset.json            # NL-ATL paired dataset
├── docs/                        # Documentation
│   ├── index.md                # Documentation hub
│   ├── quickstart.md           # Getting started guide
│   ├── installation.md         # Detailed setup
│   ├── usage.md                # CLI reference
│   ├── configuration.md        # Config file reference
│   ├── evaluation.md           # Evaluation methods
│   ├── atl_primer.md           # ATL introduction
│   ├── dataset.md              # Dataset format
│   ├── difficulty_classification.md  # Difficulty scoring
│   ├── architecture.md         # System design
│   ├── api.md                  # API reference
│   ├── genvitamin.md           # genVITAMIN integration
│   ├── slurm.md                # SLURM guide
│   └── development.md          # Contributing guide
├── tests/                       # Unit tests
├── outputs/                     # Generated outputs (created on first run)
│   ├── model_predictions/      # Model outputs
│   └── LLM-evaluation/         # Evaluation results
└── integrations/
    └── genvitamin/             # genVITAMIN model checker integration
```

## Core Workflows

### 1. Run Experiments

NL2ATL supports multiple experiment execution modes:

**Single Experiment** (quick testing):
```bash
nl2atl run-single --model qwen-3b --few_shot --seed 42
```

**Local Sweep** (multiple models/conditions on single node):
```bash
nl2atl run-all --models qwen-3b llama-8b --conditions baseline_zero_shot baseline_few_shot
```

**SLURM Array** (recommended for comprehensive sweeps):
```bash
# Preview task allocation
nl2atl run-array --count
nl2atl run-array --list-tasks

# Submit parallel jobs
sbatch scripts/slurm/submit_array.sh
```

### 2. Evaluate Results

**LLM-as-Judge Evaluation**:
```bash
# Evaluate all predictions with semantic correctness checking
nl2atl llm-judge --datasets all

# Use specific judge model
nl2atl llm-judge --datasets all --judge_model gpt-5.2

# Re-evaluate existing outputs
nl2atl llm-judge --datasets all --overwrite
```

**Inter-Rater Agreement**:
```bash
# Compute agreement metrics across judges
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets

# Include human annotations
nl2atl judge-agreement \
  --eval_dir outputs/LLM-evaluation/evaluated_datasets \
  --human_annotations path/to/human_annotations.json
```

### 3. Analyze Efficiency

```bash
# Generate comprehensive efficiency report
nl2atl model-efficiency --predictions_dir outputs/model_predictions

# Outputs:
# - efficiency_report.json (rankings and metrics)
# - efficiency_report.ipynb (interactive analysis)
```

The efficiency report provides:
- **Accuracy rankings**: Best semantic correctness
- **Cost rankings**: Most economical (USD per formula)
- **Latency rankings**: Fastest inference
- **Composite scores**: Weighted combination of accuracy, cost, and speed

### 4. Classify Dataset Difficulty

```bash
# Label dataset samples as easy or hard
nl2atl classify-difficulty --input data/dataset.json --verbose

# Customize scoring weights
nl2atl classify-difficulty \
  --input data/dataset.json \
  --formula-weight 0.4 \
  --nl-weight 0.6 \
  --threshold 5.0
```

### 5. Deploy API Service

```bash
# Start FastAPI server
uvicorn src.api_server:app --host 0.0.0.0 --port 8081

# Test endpoint
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Agent A can eventually reach goal",
    "model": "qwen-3b",
    "few_shot": true,
    "max_new_tokens": 128
  }'
```

## Output Files

All outputs are organized under `outputs/`:

```
outputs/
├── model_predictions/
│   └── <run_name>.json              # Model predictions with metadata
└── LLM-evaluation/
    ├── evaluated_datasets/
    │   └── <judge>/
    │       └── <prediction>__judge-<judge>.json
    ├── summary__judge-<judge>.json  # Judge accuracy summary
    ├── efficiency_report.json       # Cost-latency-accuracy analysis
    ├── efficiency_report.ipynb      # Interactive efficiency notebook
    └── agreement_report.json        # Inter-rater agreement metrics
```

### Prediction File Format

```json
{
  "metadata": {
    "run_id": "qwen-3b_baseline_zero_shot",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "condition": "baseline_zero_shot",
    "seed": 42,
    "total_samples": 90,
    "latency_mean_ms": 520.1,
    "timestamp_start": "2026-01-27T10:12:40Z"
  },
  "predictions": [
    {
      "id": "ex01",
      "input": "The user can guarantee...",
      "expected": "<<User>>F ticket_printed",
      "generated": "<<User>>F ticket_printed",
      "difficulty": "easy",
      "exact_match": 1,
      "latency_ms": 412.7,
      "tokens_input": 143,
      "tokens_output": 21,
      "cost_usd": 0.00234
    }
  ]
}
```

## Dataset

The NL2ATL dataset consists of 300 parallel examples of natural language requirements and their corresponding ATL formulas. Each example includes:

- **Input**: Natural language specification
- **Output**: Reference ATL formula
- **Difficulty**: Labeled as `easy` or `hard` based on formula complexity and linguistic ambiguity
- **ID**: Unique identifier

Example:
```json
{
  "id": "ex01",
  "input": "The user can guarantee that sooner or later the ticket will be printed.",
  "output": "<<User>>F ticket_printed",
  "difficulty": "easy"
}
```

Difficulty is determined by a rule-based classifier that considers:
- Formula complexity (nesting depth, operator count, coalition size)
- Natural language ambiguity (implicit operators, scope ambiguity, semantic gaps)

See [docs/dataset.md](docs/dataset.md) for complete schema and [docs/difficulty_classification.md](docs/difficulty_classification.md) for scoring details.

## Configuration

NL2ATL uses YAML configuration files for reproducible experiments:

### models.yaml

Define model registry with provider settings, LoRA parameters, and pricing:

```yaml
models:
  qwen-3b:
    name: "Qwen/Qwen2.5-3B-Instruct"
    short_name: "qwen-3b"
    provider: "huggingface"
    params_b: 3
    max_seq_length: 512
    lora_r: 64
    lora_alpha: 128
    target_modules: [q_proj, k_proj, v_proj, o_proj]
    
  gpt-5.2:
    name: "gpt-5.2"
    short_name: "gpt-5.2"
    provider: "azure"
    max_seq_length: 8192
    price_input_per_1k: 0.01
    price_output_per_1k: 0.03
```

### experiments.yaml

Configure experiment conditions, data splits, and training hyperparameters:

```yaml
experiment:
  name: "nl2atl_300_examples"
  seed: 42
  num_seeds: 5

data:
  path: "./data/dataset.json"
  test_size: 0.30
  val_size: 0.6667
  augment_factor: 10

conditions:
  - name: "baseline_zero_shot"
    finetuned: false
    few_shot: false
  - name: "finetuned_few_shot"
    finetuned: true
    few_shot: true
```

See [docs/configuration.md](docs/configuration.md) for complete reference.

## Integration with genVITAMIN

NL2ATL integrates seamlessly with genVITAMIN, an open-source model checker for multi-agent systems. The integration enables users to generate ATL formulas from natural language directly within genVITAMIN's web interface.

```bash
# Apply one-click patch to genVITAMIN backend
python integrations/genvitamin/apply_genvitamin_patch.py \
  --genvitamin-path "/path/to/genVITAMIN"

# Start NL2ATL API
uvicorn src.api_server:app --host 0.0.0.0 --port 8081

# Start genVITAMIN backend (from genVITAMIN repo)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The patch makes genVITAMIN forward natural language inputs to NL2ATL while preserving all other functionality. See [docs/genvitamin.md](docs/genvitamin.md) for detailed setup and troubleshooting.

## Evaluation Metrics

NL2ATL provides multiple evaluation approaches:

### Exact Match
- Normalized string comparison after whitespace/operator standardization
- Fast baseline metric for syntactic correctness

### LLM-as-Judge
- Semantic correctness evaluation using GPT-4/5 or other LLMs
- Provides reasoning for each judgment
- Supports multiple judge models for reliability

### Inter-Rater Agreement
- **Cohen's κ**: Pairwise agreement between judges
- **Fleiss' κ**: Agreement when all judges rate same items
- **Krippendorff's α**: Handles missing data and ordinal scales

### Efficiency Analysis
- **Accuracy**: LLM-judge or exact-match scores
- **Cost**: Per-formula USD cost from token usage
- **Latency**: Mean, P95, and throughput statistics
- **Composite score**: Normalized weighted combination

See [docs/evaluation.md](docs/evaluation.md) for detailed methodology.

## SLURM Support

For large-scale experiments with multiple models, conditions, and seeds, NL2ATL provides SLURM array job support:

**Benefits:**
- Parallel execution across GPU nodes
- Scheduler-managed resource allocation
- Fault isolation (single task failure doesn't kill sweep)
- Reproducible task mapping

**Usage:**
```bash
# Preview task allocation
nl2atl run-array --count          # Total tasks
nl2atl run-array --list-tasks     # Detailed mapping

# Filter tasks
nl2atl run-array --models qwen-3b --conditions baseline_zero_shot --count

# Submit array job
sbatch scripts/slurm/submit_array.sh
```

Each SLURM array index maps to exactly one `(seed, model, condition)` triple. Results are written independently to `outputs/model_predictions/`.

See [docs/slurm.md](docs/slurm.md) for detailed guide.

## Testing

```bash
# Run full test suite
pytest

# Run specific test
pytest tests/test_exact_match.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Getting Started
- [Installation Guide](docs/installation.md) — Setup and prerequisites
- [Quick Start](docs/quickstart.md) — First experiment in 5 minutes
- [ATL Primer](docs/atl_primer.md) — Introduction to ATL syntax and semantics

### Usage Guides
- [Usage Guide](docs/usage.md) — Complete CLI reference
- [Configuration Guide](docs/configuration.md) — YAML config reference
- [Dataset Guide](docs/dataset.md) — Dataset format and usage
- [Difficulty Classification](docs/difficulty_classification.md) — Scoring methodology
- [SLURM Guide](docs/slurm.md) — Parallel experiment execution

### Technical Reference
- [Architecture](docs/architecture.md) — System design and module layout
- [Evaluation](docs/evaluation.md) — Metrics and evaluation pipelines
- [API Reference](docs/api.md) — Public modules and classes

### Integration
- [genVITAMIN Integration](docs/genvitamin.md) — Model checker integration guide

### Contributing
- [Development Guide](docs/development.md) — Extending and contributing

## Research Paper

If you use NL2ATL in your research, please cite:

```bibtex
@inproceedings{aruta2026nl2atl,
  title={Translating Natural Language to Strategic Temporal Specifications via LLMs},
  author={Aruta, Marco and Improta, Francesco and Malvone, Vadim and Murano, Aniello and Perli{\'c}, Vladana},
  booktitle={[Conference/Journal]},
  year={2026}
}
```

## License

NL2ATL is released under the [MIT License](https://opensource.org/licenses/MIT). See [LICENSE](LICENSE) for details.

## Acknowledgments

This research was conducted at:
- **Telecom Paris**
- **STMicroelectronics Grenoble**
- **University of Naples Federico II**

## Contact & Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/vladanaSTM/nl2atl/issues)
- **Documentation**: Full documentation at [docs/index.md](docs/index.md)
- **Repository**: https://github.com/vladanaSTM/nl2atl

## FAQ

**Q: What models are supported?**  
A: Any Hugging Face model with instruction-following capabilities (Qwen, Llama, Mistral, etc.) and Azure OpenAI models.

**Q: Do I need a GPU?**  
A: GPUs are recommended for fine-tuning and local inference with larger models. Azure API can be used without local GPU resources.

**Q: Can I use my own dataset?**  
A: Yes! The dataset format is simple JSON. See [docs/dataset.md](docs/dataset.md) for schema details.

**Q: How do I add a new model?**  
A: Add an entry to `configs/models.yaml` and ensure the provider (Hugging Face or Azure) is properly configured.

**Q: What if I don't have SLURM?**  
A: Use `nl2atl run-all` for local sweeps or `nl2atl run-single` for individual experiments.

**Q: How are costs calculated?**  
A: Costs are derived from token usage and per-1k pricing in `configs/models.yaml`. For local GPUs, you can specify `gpu_hour_usd` for estimation.

**Q: Can I integrate NL2ATL into my own tools?**  
A: Yes! Use the FastAPI service (`src/api_server.py`) or import modules directly. See [docs/api.md](docs/api.md) for details.

---

**Ready to get started?** Head to [docs/quickstart.md](docs/quickstart.md) for a 5-minute walkthrough!
