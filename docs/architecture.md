# Architecture

This document provides a concise overview of NL2ATL's architecture, module organization, and data flow as implemented in the current codebase.

## System Overview

NL2ATL uses a layered structure with a consolidated CLI that dispatches to task-specific subcommands.

```mermaid
graph TB
    subgraph "Interface Layer"
        CLI[CLI Commands]
        CFG[configs/*.yaml]
    end

    subgraph "Orchestration Layer"
        RUN[ExperimentRunner]
        REP[ExperimentReporter]
        DATA[ExperimentDataManager]
    end

    subgraph "Core Layer"
        MOD[Model Registry]
        PROMPT[Few-shot Prompting]
        EVAL[ExactMatchEvaluator]
    end

    subgraph "Evaluation Layer"
        LJ[LLM Judge Pipeline]
        AGREE[Judge Agreement]
        DIFF[Difficulty Classifier]
    end

    subgraph "Infrastructure Layer"
        AZURE[Azure Client]
        IO[I/O Utilities]
        ENV[Environment]
    end

    CLI --> RUN
    CFG --> RUN
    RUN --> DATA
    RUN --> MOD
    RUN --> PROMPT
    RUN --> EVAL
    RUN --> REP
    LJ --> AZURE
    DATA --> IO
    REP --> IO
    AZURE --> ENV
```

## Execution Workflows

### Experiment Workflow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Runner
    participant DataMgr
    participant Registry
    participant Evaluator
    participant Reporter

    User->>CLI: nl2atl run-all / run-single
    CLI->>Runner: Config.from_yaml(models.yaml, experiments.yaml)
    Runner->>DataMgr: prepare_data()
    DataMgr-->>Runner: train_aug, val, test
    Runner->>Registry: load_model(...)
    Runner->>Evaluator: evaluate(model, test_data)
    Evaluator-->>Runner: metrics + per-sample results
    Runner->>Reporter: save_result(run_name, result)
    Reporter-->>CLI: output paths
```

### LLM Judge Workflow

```mermaid
sequenceDiagram
    participant CLI
    participant Judge
    participant Client
    participant Metrics

    CLI->>Judge: evaluate_prediction_file(...)
    loop For each prediction
        Judge->>Client: complete(prompt)
        Client-->>Judge: response
    end
    Judge->>Metrics: compute_metrics(rows)
```

## Package Structure

```
src/
├── cli/                    # Command-Line Interface
│   ├── main.py             # Command dispatcher
│   ├── run_all_experiments.py
│   ├── run_single_experiment.py
│   ├── run_llm_judge.py
│   ├── run_judge_agreement.py
│   └── classify_difficulty.py
│
├── experiment/             # Experiment Orchestration
│   ├── runner.py           # Core experiment workflow
│   ├── data_manager.py     # Dataset handling
│   └── reporter.py         # Logging and output
│
├── models/                 # Model Management
│   ├── registry.py         # Model loading and caching
│   ├── utils.py            # Model utilities
│   └── few_shot.py         # Few-shot prompting
│
├── evaluation/             # Evaluation Framework
│   ├── base.py             # Abstract evaluator interface
│   ├── exact_match.py      # Exact-match evaluation
│   ├── difficulty.py       # Difficulty classification
│   ├── judge_agreement.py  # Inter-rater metrics
│   ├── model_efficiency.py # Cost/latency/accuracy trade-off report
│   └── llm_judge/          # LLM-as-a-Judge Pipeline
│       ├── client.py       # LLM client wrappers
│       ├── prompts.py      # Prompt templates
│       ├── parser.py       # Response parsing
│       ├── pipeline.py     # Orchestration
│       └── metrics.py      # Metric computation
│
├── infra/                  # Infrastructure
│   ├── azure.py            # Azure OpenAI client
│   ├── io.py               # File I/O utilities
│   └── env.py              # Environment management
│
├── config.py               # Configuration schema
└── constants.py            # Shared constants
```

## Data Flow

### Prediction File Structure

Experiments write JSON files in `outputs/model_predictions/` with:

- `metadata`: run details (model, condition, seed, timing)
- `predictions`: list of evaluated samples

### LLM Judge Outputs

The LLM judge writes evaluated datasets to `outputs/LLM-evaluation/evaluated_datasets/<judge>/` and a summary JSON plus an optional notebook.

### Model Efficiency Outputs

The efficiency report aggregates metrics from prediction files and (optionally) LLM judge summaries, producing:

- `outputs/LLM-evaluation/efficiency_report.json`
- `outputs/LLM-evaluation/efficiency_report.ipynb`

## Extensibility Points

- Add new models by extending `configs/models.yaml` and, if needed, provider logic in `src/models/registry.py`.
- Add new evaluators by implementing `BaseEvaluator` in `src/evaluation/`.
- Add new CLI tasks by adding a `src/cli/*.py` entry and registering it in `src/cli/main.py`.
    end
    
    subgraph "Output"
        ATL["ATL* Formula<br/>⟨⟨A,B⟩⟩ F win"]
    end
    
    NL --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> ATL
```

### Artifact Relationships

```mermaid
erDiagram
    EXPERIMENT ||--o{ PREDICTION : produces
    EXPERIMENT {
        string id
        string model_id
        string config_hash
        datetime timestamp
        int seed
    }
    
    PREDICTION ||--o{ EVALUATION : evaluated_by
    PREDICTION {
        string id
        string experiment_id
        string input_nl
        string output_atl
        string reference_atl
        string difficulty
    }
    
    EVALUATION {
        string id
        string prediction_id
        string judge_model
        string verdict
        float confidence
        string reasoning
    }
    
    DATASET ||--o{ PREDICTION : source_of
    DATASET {
        string id
        int total_samples
        int easy_count
        int hard_count
    }
```

---

## Evaluation Pipeline

### Evaluation Methods Hierarchy

```mermaid
classDiagram
    class BaseEvaluator {
        <<abstract>>
        +evaluate(predictions, references) Dict
        +evaluate_single(prediction, reference) Dict
    }
    
    class ExactMatchEvaluator {
        +normalize(text) str
        +evaluate(predictions, references) Dict
        +evaluate_single(prediction, reference) Dict
    }
    
    class LLMJudgeEvaluator {
        -client: JudgeClient
        +evaluate(predictions, references) Dict
        +evaluate_single(prediction, reference) Dict
    }
    
    class DifficultyClassifier {
        +classify(sample) str
        +classify_batch(samples) List
    }
    
    BaseEvaluator <|-- ExactMatchEvaluator
    BaseEvaluator <|-- LLMJudgeEvaluator
    BaseEvaluator <|-- DifficultyClassifier
    
    class JudgeClient {
        <<interface>>
        +complete(prompt) str
    }
    
    class AzureJudgeClient {
        +complete(prompt) str
    }
    
    JudgeClient <|.. AzureJudgeClient
    LLMJudgeEvaluator --> JudgeClient
```

### Evaluation for ATL* Translation

The evaluation considers ATL*-specific criteria:

```mermaid
flowchart TD
    subgraph "Evaluation Criteria"
        C1[Coalition Correctness<br/>Correct agents identified?]
        C2[Operator Mapping<br/>Correct temporal operators?]
        C3[Structural Validity<br/>Well-formed formula?]
        C4[Semantic Equivalence<br/>Same meaning as reference?]
    end
    
    subgraph "Verdict"
        V1[Correct]
        V2[Partial]
        V3[Incorrect]
    end
    
    C1 --> V1
    C2 --> V1
    C3 --> V1
    C4 --> V1
    
    C1 --> V2
    C2 --> V2
    
    C1 --> V3
    C3 --> V3
```

---

## Extensibility Points

### Adding New Components

```mermaid
flowchart TD
    subgraph "Extension Points"
        E1[New Model Backend<br/>e.g., Ollama, HuggingFace]
        E2[New Evaluator<br/>e.g., Syntax Checker]
        E3[New CLI Command]
        E4[New Data Source]
    end
    
    subgraph "Implementation Steps"
        S1[Add to models/registry.py<br/>Update models.yaml]
        S2[Extend evaluation/base.py<br/>Add to evaluation/]
        S3[Create cli/run_*.py<br/>Register in cli/main.py]
        S4[Extend experiment/data_manager.py]
    end
    
    E1 --> S1
    E2 --> S2
    E3 --> S3
    E4 --> S4
```

### Model Backend Extension

```mermaid
classDiagram
    class ModelRegistry {
        -_cache: Dict
        +load(config) Model
        +unload(model_id)
        +register_backend(name, loader)
    }
    
    class AzureBackend {
        +load(config) AzureModel
    }
    
    class LocalBackend {
        +load(config) LocalModel
    }
    
    class OllamaBackend {
        +load(config) OllamaModel
    }
    
    class HuggingFaceBackend {
        +load(config) HFModel
    }
    
    ModelRegistry --> AzureBackend : uses
    ModelRegistry --> LocalBackend : uses
    ModelRegistry --> OllamaBackend : future
    ModelRegistry --> HuggingFaceBackend : future
```

---

## Output Structure

### Directory Organization

```
outputs/
├── model_predictions/
│   ├── {model}_{timestamp}_{seed}.json
│   └── ...
└── LLM-evaluation/
    └── evaluated_datasets/
        └── {judge_model}/
            ├── {experiment_id}_verdicts.json
            ├── {experiment_id}_metrics.json
            └── agreement_report.json
```

### Prediction Artifact Schema

```json
{
  "metadata": {
    "experiment_id": "qwen32b_fewshot_20240115",
    "model_id": "qwen-32b",
    "timestamp": "2024-01-15T10:30:00Z",
    "seed": 42,
    "few_shot_count": 3
  },
  "predictions": [
    {
      "id": "sample_001",
      "input_nl": "The user can guarantee that...",
      "output_atl": "⟨⟨User⟩⟩ X ¬timeout",
      "reference_atl": "⟨⟨User⟩⟩ X ¬timeout",
      "difficulty": "easy"
    }
  ]
}
```
