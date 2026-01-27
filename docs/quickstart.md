# Quick Start Guide

This guide will walk you through running your first NL2ATL experiment in under 10 minutes. By the end, you'll have generated ATL formulas, evaluated them, and analyzed the results.

## Prerequisites

Before starting, ensure you've completed the [Installation Guide](installation.md):

✅ Python 3.10+ installed  
✅ Repository cloned and dependencies installed  
✅ Virtual environment activated  
✅ `.env` file configured (if using Azure models)

## What You'll Do

In this quickstart, you'll:

1. ✨ Generate ATL formulas using a pretrained model
2. 📊 Evaluate predictions with exact-match and LLM judge
3. 💰 Analyze model efficiency (cost, latency, accuracy)
4. 🔍 Explore the dataset and outputs

**Time**: ~10 minutes (depending on model inference speed)

---

## Step 1: Verify Installation

First, confirm the CLI is properly installed:

```bash
nl2atl --help
```

Expected output:
```
Usage: nl2atl [OPTIONS] COMMAND [ARGS]...

Commands:
  run-all              Run multiple experiments
  run-single           Run a single experiment
  run-array            Run SLURM array task
  llm-judge            Evaluate with LLM judge
  judge-agreement      Compute inter-rater agreement
  model-efficiency     Generate efficiency report
  classify-difficulty  Label dataset difficulty
```

If you see this, you're ready to proceed! 🎉

---

## Step 2: Inspect the Dataset

Let's look at the training data to understand what we're working with:

```bash
# View first 3 examples
python -c "
from src.infra.io import load_json
dataset = load_json('data/dataset.json')[:3]
for item in dataset:
    print(f\"ID: {item['id']}\")
    print(f\"Input: {item['input']}\")
    print(f\"Output: {item['output']}\")
    print(f\"Difficulty: {item['difficulty']}\")
    print('-' * 60)
"
```

**Example output:**
```
ID: ex01
Input: The user can guarantee that sooner or later the ticket will be printed.
Output: <<User>>F ticket_printed
Difficulty: easy
------------------------------------------------------------
ID: ex02
Input: The machine can guarantee that if the payment has been completed, 
       then sooner or later the gate will open.
Output: <<Machine>>G (paid -> F gate_open)
Difficulty: easy
------------------------------------------------------------
```

**What you're seeing:**
- **Input**: Natural language requirement
- **Output**: Ground-truth ATL formula
- **Difficulty**: Classification based on complexity

The dataset has 300 examples split 70/30 for train/test with stratified sampling by difficulty.

---

## Step 3: Run Your First Experiment

Now let's generate ATL formulas using a pretrained model with few-shot prompting:

### Option A: Local Inference (Recommended for First Run)

```bash
nl2atl run-single --model qwen-3b --few_shot
```

**What's happening:**
1. Loads the Qwen 2.5 3B Instruct model from Hugging Face
2. Prepares test set (70/30 split, stratified by difficulty)
3. Generates ATL formulas for each test example
4. Computes exact-match accuracy
5. Saves predictions to `outputs/model_predictions/`

**Expected output:**
```
Loading model: Qwen/Qwen2.5-3B-Instruct
Preparing dataset...
  Train: 210 examples
  Validation: 0 examples (validation disabled)
  Test: 90 examples

Running inference on test set...
Progress: 90/90 [100%] ⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿ 2:15

Results:
  Total samples: 90
  Successful: 90
  Failed: 0
  Exact match accuracy: 0.82
  Mean latency: 520.1 ms
  P95 latency: 910.5 ms

Predictions saved to: outputs/model_predictions/qwen-3b_baseline_few_shot.json
```

**Runtime**: ~2-5 minutes on GPU, ~10-20 minutes on CPU

### Option B: Azure OpenAI (Faster if Configured)

If you have Azure OpenAI configured:

```bash
nl2atl run-single --model gpt-5.2 --few_shot
```

This uses cloud inference and is typically faster, but requires API credits.

---

## Step 4: Inspect Predictions

Let's examine what the model generated:

```bash
# View first prediction
python -c "
from src.infra.io import load_json
result = load_json('outputs/model_predictions/qwen-3b_baseline_few_shot.json')
pred = result['predictions'][0]
print('Input:', pred['input'])
print('Expected:', pred['expected'])
print('Generated:', pred['generated'])
print('Match:', '✓' if pred['exact_match'] else '✗')
print('Latency:', f\"{pred['latency_ms']:.1f} ms\")
"
```

**Example output:**
```
Input: The user can guarantee that sooner or later the ticket will be printed.
Expected: <<User>>F ticket_printed
Generated: <<User>>F ticket_printed
Match: ✓
Latency: 412.7 ms
```

**Prediction file structure:**
```json
{
  "metadata": {
    "run_id": "qwen-3b_baseline_few_shot",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "total_samples": 90,
    "exact_match_accuracy": 0.82,
    "latency_mean_ms": 520.1
  },
  "predictions": [
    {
      "id": "ex01",
      "input": "...",
      "expected": "<<User>>F ticket_printed",
      "generated": "<<User>>F ticket_printed",
      "exact_match": 1,
      "latency_ms": 412.7,
      "tokens_input": 143,
      "tokens_output": 21
    }
  ]
}
```

---

## Step 5: Evaluate with LLM Judge

Exact-match is strict—semantically correct formulas might fail due to formatting. Let's use an LLM judge for semantic evaluation:

```bash
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot.json
```

**What's happening:**
1. Loads your predictions
2. For each prediction, asks an LLM: "Is this formula semantically correct?"
3. LLM responds with `{"correct": "yes"|"no", "reasoning": "..."}`
4. Computes LLM-judge accuracy and saves detailed results

**Expected output:**
```
Evaluating qwen-3b_baseline_few_shot.json with judge gpt-5.2
Progress: 90/90 [100%] ⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿ 1:30

Results:
  Exact matches (skipped LLM): 74/90 (82.2%)
  LLM evaluated: 16/90 (17.8%)
  LLM correct: 8/16 (50.0%)
  Total correct: 82/90 (91.1%)

Saved to: outputs/LLM-evaluation/evaluated_datasets/gpt-5.2/
         qwen-3b_baseline_few_shot__judge-gpt-5.2.json
```

**Interpretation:**
- **82.2% exact match** → 74 formulas matched character-for-character
- **50% LLM correct** → Half of the 16 non-matches were still semantically correct
- **91.1% total accuracy** → Real model performance after accounting for valid variations

---

## Step 6: Compare Multiple Models (Optional)

Want to compare models? Run multiple experiments:

```bash
# Run zero-shot and few-shot for comparison
nl2atl run-single --model qwen-3b --few_shot
nl2atl run-single --model qwen-3b  # zero-shot

# Evaluate both
nl2atl llm-judge --datasets all
```

---

## Step 7: Generate Efficiency Report

Now let's analyze cost-latency-accuracy trade-offs:

```bash
nl2atl model-efficiency --predictions_dir outputs/model_predictions
```

**What's happening:**
1. Aggregates metrics across all prediction files
2. Computes costs from token usage (if pricing configured)
3. Calculates normalized composite scores
4. Generates rankings and comparison report

**Output files:**
- `outputs/LLM-evaluation/efficiency_report.json` — Structured data
- `outputs/LLM-evaluation/efficiency_report.ipynb` — Interactive notebook

**Example report excerpt:**
```json
{
  "overall_stats": {
    "total_runs": 2,
    "total_formulas": 180,
    "models_tested": ["qwen-3b"]
  },
  "rankings": {
    "by_accuracy": [
      {
        "run_id": "qwen-3b_baseline_few_shot",
        "accuracy": 0.911,
        "rank": 1
      },
      {
        "run_id": "qwen-3b_baseline_zero_shot",
        "accuracy": 0.821,
        "rank": 2
      }
    ],
    "by_latency": [
      {
        "run_id": "qwen-3b_baseline_zero_shot",
        "latency_mean_ms": 380.2,
        "rank": 1
      }
    ]
  }
}
```

**Interpretation:**
- Few-shot is more accurate (91.1% vs 82.1%)
- Zero-shot is faster (380ms vs 520ms per formula)
- Choose based on your priority: accuracy or speed

---

## Step 8: Explore Advanced Features

### Classify Dataset Difficulty

See how complex each example is:

```bash
nl2atl classify-difficulty --input data/dataset.json --verbose
```

This adds `difficulty_scores` to each example showing formula complexity and NL ambiguity.

### Run Multiple Conditions

Test all combinations of fine-tuning and few-shot:

```bash
nl2atl run-all --models qwen-3b --conditions baseline_zero_shot baseline_few_shot
```

### Compute Inter-Rater Agreement

If you have multiple judge outputs:

```bash
nl2atl judge-agreement --eval_dir outputs/LLM-evaluation/evaluated_datasets
```

This computes Cohen's κ, Fleiss' κ, and Krippendorff's α.

---

## Step 9: Try the API Service

Start NL2ATL as a REST API:

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8081
```

Test it with curl:

```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Agent A can eventually reach the goal",
    "model": "qwen-3b",
    "few_shot": true,
    "max_new_tokens": 128
  }'
```

**Expected response:**
```json
{
  "formula": "<<A>>F goal",
  "model_key": "qwen-3b",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "provider": "huggingface",
  "latency_ms": 425.3
}
```

---

## What You've Accomplished ✨

Congratulations! You've:

✅ Generated ATL formulas from natural language  
✅ Evaluated predictions with exact-match and LLM judge  
✅ Analyzed model efficiency and trade-offs  
✅ Explored the dataset structure  
✅ Tested the API service

## Next Steps

### For Researchers

1. **Run comprehensive experiments** → [SLURM Guide](slurm.md)
2. **Fine-tune models** → [Configuration Guide](configuration.md)
3. **Compare multiple models** → `nl2atl run-all`
4. **Analyze results** → [Evaluation Guide](evaluation.md)

### For Developers

1. **Integrate NL2ATL** → [API Reference](api.md)
2. **Extend the framework** → [Development Guide](development.md)
3. **Connect to genVITAMIN** → [genVITAMIN Integration](genvitamin.md)

### For Understanding ATL

1. **Learn ATL syntax** → [ATL Primer](atl-primer.md)
2. **Explore dataset** → [Dataset Guide](dataset.md)
3. **Understand difficulty** → [Difficulty Classification](difficulty_classification.md)

## Troubleshooting

### Model loading errors

**Problem**: `ModuleNotFoundError: transformers`  
**Solution**: Ensure dependencies are installed: `pip install -r requirements.txt`

### CUDA out of memory

**Problem**: GPU runs out of memory during inference  
**Solution**: Add `load_in_4bit: true` to model config in `configs/models.yaml`

### Azure authentication errors

**Problem**: `AuthenticationError: Invalid API key`  
**Solution**: Check `.env` has correct `AZURE_API_KEY` and `AZURE_INFER_ENDPOINT`

### Slow inference

**Problem**: CPU inference is very slow  
**Solution**: 
- Use smaller model (1-3B parameters)
- Use Azure API instead
- Run on GPU if available

### Can't find predictions

**Problem**: `outputs/model_predictions/` is empty  
**Solution**: Ensure you ran `nl2atl run-single` or `nl2atl run-all` first

## Common CLI Patterns

```bash
# Single experiment, specific model
nl2atl run-single --model qwen-3b --few_shot --seed 42

# Sweep all models with one condition
nl2atl run-all --conditions baseline_few_shot

# Sweep one model with all conditions
nl2atl run-all --models qwen-3b

# Evaluate specific predictions
nl2atl llm-judge --datasets qwen-3b_baseline_few_shot.json

# Evaluate all predictions
nl2atl llm-judge --datasets all

# Force re-evaluation
nl2atl llm-judge --datasets all --overwrite

# Generate efficiency report with custom weights
nl2atl model-efficiency \
  --predictions_dir outputs/model_predictions \
  --weight_accuracy 0.5 \
  --weight_cost 0.3 \
  --weight_latency 0.2
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Run single experiment | `nl2atl run-single --model qwen-3b --few_shot` |
| Run experiment sweep | `nl2atl run-all --models qwen-3b` |
| Evaluate predictions | `nl2atl llm-judge --datasets all` |
| Efficiency analysis | `nl2atl model-efficiency --predictions_dir outputs/model_predictions` |
| Classify difficulty | `nl2atl classify-difficulty --input data/dataset.json` |
| Start API | `uvicorn src.api_server:app --port 8081` |
| View help | `nl2atl --help` |
| View command help | `nl2atl run-single --help` |

## Learning Resources

- **ATL Introduction**: [ATL Primer](atl-primer.md)
- **Full CLI Reference**: [Usage Guide](usage.md)
- **All Evaluation Methods**: [Evaluation Guide](evaluation.md)
- **Configuration Options**: [Configuration Guide](configuration.md)
- **System Architecture**: [Architecture](architecture.md)

---

**Questions?** Check the [full documentation](index.md) or [open an issue](https://github.com/vladanaSTM/nl2atl/issues).

**Ready for more?** → [Usage Guide](usage.md) for complete CLI reference
