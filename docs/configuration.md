# Configuration

NL2ATL is configured via two YAML files:

- configs/models.yaml
- configs/experiments.yaml

## models.yaml

Defines model metadata, providers, and endpoints. Each model entry is keyed by a short identifier used in CLI commands.

## experiments.yaml

Defines dataset paths, training hyperparameters, and experiment conditions. Key sections:

- experiment: seed, seeds, num_seeds
- data: path, test_size, val_size, augment_factor
- training: num_epochs, batch_size, learning_rate, etc.
- few_shot: num_examples
- wandb: project, entity
- conditions: list of experiment condition definitions

## Environment Variables

Used for Azure judging:

- AZURE_API_KEY
- AZURE_INFER_ENDPOINT
- AZURE_API_VERSION (optional)
- AZURE_USE_CACHE (optional)
- AZURE_VERIFY_SSL (optional)
