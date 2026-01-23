# NL2ATL Documentation

> **Natural Language to ATL**: A research framework for translating natural language specifications into Alternating-Time Temporal Logic formulas using Large Language Models.

## Overview

NL2ATL is a Python framework developed to support the research presented in:

> **Translating Natural Language to Strategic Temporal Specifications via LLMs**
>
> *M. Aruta, F. Improta, V. Malvone, A. Murano, V. Perlić*
>
> University of Naples Federico II & Telecom Paris

The framework enables automatic translation of natural language requirements involving strategic reasoning into well-formed ATL formulas (coalitions + temporal operators). It provides tooling for experiments, evaluation, and dataset difficulty classification.

## Motivation

Formal verification of Multi-Agent Systems (MAS) relies on logical formalisms capable of expressing strategic abilities of interacting agents. Strategic logics such as ATL and ATL* allow reasoning about the existence of joint strategies for coalitions of agents under adversarial conditions. NL2ATL bridges this gap by enabling non-expert users to specify strategic properties via natural language.

## Documentation

### Getting Started

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Setup instructions and requirements |
| [Quick Start](quickstart.md) | Run your first experiment |
| [ATL* Primer](atl-primer.md) | Introduction to Alternating-Time Temporal Logic |

### User Guide

| Document | Description |
|----------|-------------|
| [Usage Guide](usage.md) | CLI commands and experiment workflows |
| [Configuration](configuration.md) | Config file formats and options |
| [Dataset](dataset.md) | Dataset structure and format |

### Technical Reference

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, module structure, and data flow |
| [Evaluation](evaluation.md) | Evaluation methods and metrics |
| [API Reference](api.md) | Module and class documentation |

### Contributing

| Document | Description |
|----------|-------------|
| [Development](development.md) | Contributing and extending the framework |

## Key Features

- **Strategic Logic Support**: Native handling of ATL coalition modalities and temporal operators.
- **Multiple Evaluation Methods**: Exact-match scoring and LLM-as-a-judge evaluation.
- **Reproducibility**: Experiment tracking with W&B integration and seed aggregation.
- **Extensible Design**: Modular model registry, evaluators, and CLI subcommands.

## Project Structure

```
nl2atl/
├── nl2atl.py              # CLI entrypoint
├── configs/               # Experiment and model configurations
├── data/                  # Dataset (NL-ATL pairs)
├── outputs/               # Predictions and evaluations
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── src/                   # Source code
  ├── cli/               # Command-line interface
  ├── experiment/        # Experiment orchestration
  ├── models/            # Model loading and inference
  ├── evaluation/        # Evaluation pipelines
  └── infra/             # Infrastructure utilities
```

## Citation

If you use NL2ATL in your research, please cite our paper:

```bibtex
@inproceedings{aruta2024nl2atl,
  title={Translating Natural Language to Strategic Temporal Specifications via LLMs},
  author={Aruta, Marco and Improta, Francesco and Malvone, Vadim and Murano, Aniello and Perli{\'c}, Vladana},
  booktitle={[Conference/Journal]},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- University of Naples Federico II
- Telecom Paris