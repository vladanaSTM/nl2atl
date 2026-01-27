# API Service

NL2ATL exposes a lightweight FastAPI service for NL→ATL generation.

## Start the server

Run from the repo root so config paths resolve:

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

If you run elsewhere, set absolute paths:

```bash
NL2ATL_MODELS_CONFIG=/abs/path/to/nl2atl/configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=/abs/path/to/nl2atl/configs/experiments.yaml
```

## Endpoints

### Health

```bash
curl http://localhost:8081/health
```

### Generate

```bash
curl -X POST http://localhost:8081/generate \
    -H "Content-Type: application/json" \
    -d '{
        "description": "Agent A can eventually reach goal",
        "model": "qwen-3b",
        "few_shot": true,
        "max_new_tokens": 128
    }'
```

## Request fields

- `description` (string, required): NL requirement
- `model` (string, optional): model key from `configs/models.yaml`
- `few_shot` (bool, optional): enable few‑shot prompt
- `num_few_shot` (int, optional): override few‑shot count
- `adapter` (string, optional): LoRA adapter name/path (HuggingFace only)
- `max_new_tokens` (int, optional): generation limit
- `return_raw` (bool, optional): include raw output

## Response fields

- `formula`: generated ATL formula
- `model_key`: resolved model key
- `model_name`: resolved model name
- `provider`: `huggingface` or `azure`
- `latency_ms`: end‑to‑end latency

For UI wiring, see [integrations/genvitamin.md](integrations/genvitamin.md).

Generates Cohen’s κ, Fleiss’ κ, and Krippendorff’s α reports from evaluated datasets.

### `src.evaluation.model_efficiency`

Key functions:

- `build_efficiency_report(...)`
- `build_efficiency_notebook(...)`

### `src.evaluation.judge_agreement.generate_agreement_report_with_human`

Same as `generate_agreement_report`, but merges a human annotation JSON as an additional judge.

### `src.evaluation.llm_judge`

Key classes and functions:

- `LLMJudge`
- `LLMJudgeEvaluator`
- `evaluate_prediction_file(...)`
- `compute_metrics(...)`
- `build_summary(...)`
- `build_summary_notebook(...)`
- `run_llm_judge(...)`

Data classes:

- `JudgeDecision`
- `JudgePromptConfig`
- `JudgeVerdict`
- `JudgeMetrics`

---

## Infrastructure Module

### `src.infra.azure`

- `AzureConfig.from_env()` and `AzureConfig.from_env_optional()`
- `AzureClient.generate(prompt, max_new_tokens=256, return_usage=False)`
- `GenerationResult` dataclass (text + usage token counts)

### `src.infra.io`

- `load_yaml(path)`
- `load_json(path)`
- `save_json(data, path)`
- `load_json_safe(path, default=None)`

### `src.infra.env`

- `load_env()`
```python
def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with UTF-8 encoding."""
    
def save_json(
    data: Dict[str, Any], 
    path: Path, 
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with safe loading."""
    
def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save data to YAML file."""
```

---


```python
from enum import Enum

class Provider(Enum):
    """Model provider types."""
    AZURE = "azure"
    LOCAL = "local"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

class Verdict(Enum):
    """Judge verdict types."""
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"

class Difficulty(Enum):
    """Sample difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ATL* Operators
ATL_OPERATORS = {"G", "F", "X", "U", "R", "W"}

# Default paths
DEFAULT_DATA_PATH = Path("data/dataset.json")
DEFAULT_OUTPUT_DIR = Path("outputs/model_predictions")
DEFAULT_EVAL_DIR = Path("outputs/LLM-evaluation")

# Default parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = 42
```
