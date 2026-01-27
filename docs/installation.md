# Installation Guide

This guide will help you set up NL2ATL for research and development. Follow these steps to install dependencies, configure your environment, and verify the installation.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: 8 GB (16 GB+ recommended for local inference)
- **Storage**: 10 GB free space (more for model weights)
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements

- **Python**: 3.10 or 3.11
- **RAM**: 16 GB+ (32 GB for larger models)
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for local fine-tuning/inference)
  - CUDA 11.8+ for optimal PyTorch performance
  - RTX 3090, RTX 4090, A100, or similar recommended
- **Storage**: 50 GB+ (for models, datasets, outputs)
- **OS**: Ubuntu 20.04+ or similar Linux distribution

### Optional Requirements

- **SLURM**: For parallel experiment execution across GPU nodes
- **Weights & Biases account**: For experiment tracking
- **Azure OpenAI subscription**: For cloud inference and judge models
- **Hugging Face account**: For accessing gated models

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

This method installs all dependencies and sets up the CLI.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
```

#### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Verify activation:**
```bash
which python  # Should point to .venv/bin/python
```

#### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install NL2ATL as editable package (enables 'nl2atl' command)
pip install -e .
```

**What gets installed:**
- PyTorch (with CUDA support if available)
- Transformers (Hugging Face models)
- FastAPI (API service)
- Click (CLI framework)
- Pytest (testing)
- And other required packages

#### Step 4: Verify Installation

```bash
# Check CLI is available
nl2atl --help

# Check Python imports work
python -c "from src.experiment import ExperimentRunner; print('✓ Imports OK')"

# Check CUDA availability (optional, for GPU users)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```
Usage: nl2atl [OPTIONS] COMMAND [ARGS]...

  NL2ATL: Natural Language to ATL formula generation

Commands:
  run-all              Run multiple experiments
  run-single           Run a single experiment
  ...
```

---

### Method 2: Development Installation

For contributors or those extending NL2ATL.

```bash
# Clone repository
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Verify tests pass
pytest -v
```

---

### Method 3: Minimal Installation (API Only)

If you only need the API service without experiment infrastructure:

```bash
# Clone repository
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install minimal dependencies
pip install fastapi uvicorn transformers torch python-dotenv pyyaml

# Install NL2ATL
pip install -e .
```

---

## Configuration

### Step 1: Create Environment File

```bash
cp .env.example .env
```

### Step 2: Configure Environment Variables

Edit `.env` with your preferred text editor and set the variables you need:

#### Core Variables

```bash
# Python path (if imports fail)
PYTHONPATH=.

# Random seed for reproducibility
SEED=42
```

#### Azure OpenAI (Optional)

If using Azure-hosted models for inference or as judge:

```bash
AZURE_API_KEY=your_azure_api_key_here
AZURE_INFER_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_INFER_MODEL=gpt-5.2
AZURE_API_VERSION=2024-08-01-preview
AZURE_USE_CACHE=true
AZURE_VERIFY_SSL=false  # Set to true in production
```

**Where to find these:**
1. Log into [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Copy "Endpoint" → `AZURE_INFER_ENDPOINT`
4. Navigate to "Keys and Endpoint"
5. Copy "Key 1" → `AZURE_API_KEY`
6. Note your deployment name → `AZURE_INFER_MODEL`

#### Hugging Face (Optional)

For accessing gated models (Llama, etc.):

```bash
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

**How to get token:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Navigate to Settings → Access Tokens
3. Create a token with "Read" permissions
4. Copy token value

#### Weights & Biases (Optional)

For experiment tracking and visualization:

```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=nl2atl
WANDB_ENTITY=your_wandb_username
```

**How to get API key:**
1. Create account at [wandb.ai](https://wandb.ai)
2. Navigate to Settings → API Keys
3. Copy key value

#### API Service (Required for API Deployment)

```bash
NL2ATL_DEFAULT_MODEL=qwen-3b
NL2ATL_MODELS_CONFIG=configs/models.yaml
NL2ATL_EXPERIMENTS_CONFIG=configs/experiments.yaml
```

**Note:** If running API from outside repo root, use absolute paths.

### Step 3: Verify Configuration

```bash
# Check environment variables loaded
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Azure configured:', os.getenv('AZURE_API_KEY') is not None)
print('HF configured:', os.getenv('HUGGINGFACE_TOKEN') is not None)
print('W&B configured:', os.getenv('WANDB_API_KEY') is not None)
"
```

---

## Platform-Specific Setup

### Linux (Ubuntu/Debian)

Most straightforward platform for NL2ATL.

```bash
# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# CUDA (for GPU users)
# Follow NVIDIA's official guide for your distribution
# https://developer.nvidia.com/cuda-downloads

# Continue with standard installation...
```

### macOS

```bash
# Ensure Python 3.10+ installed
brew install python@3.10

# Use python3.10 explicitly
python3.10 -m venv .venv
source .venv/bin/activate

# Continue with standard installation...
```

**Note:** MPS (Metal Performance Shaders) is supported by PyTorch for Apple Silicon, but CUDA-specific features won't work.

### Windows

**Recommended:** Use WSL2 (Windows Subsystem for Linux) for best compatibility.

#### Option A: WSL2 (Recommended)

```bash
# Install WSL2 and Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Linux installation steps
```

#### Option B: Native Windows

```powershell
# Ensure Python 3.10+ installed
python --version

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# If execution policy blocked, run as admin:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### SLURM Cluster

For running experiments on HPC clusters:

```bash
# Load required modules (example, adjust for your cluster)
module load python/3.10
module load cuda/11.8
module load gcc/9.3.0

# Create virtual environment in your home directory
python -m venv ~/envs/nl2atl
source ~/envs/nl2atl/bin/activate

# Clone and install
git clone https://github.com/vladanaSTM/nl2atl.git
cd nl2atl
pip install -r requirements.txt
pip install -e .

# Test on login node (no GPU required for this test)
nl2atl --help
```

See [SLURM Guide](slurm.md) for detailed cluster setup.

---

## GPU Setup

### NVIDIA GPU (CUDA)

```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# Expected output:
# CUDA available: True
# CUDA version: 11.8
# GPU count: 1
```

If CUDA is not available but you have an NVIDIA GPU:

1. Install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
2. Install CUDA Toolkit from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
3. Reinstall PyTorch with CUDA support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPU (ROCm)

PyTorch with ROCm support is available but less tested:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### Apple Silicon (M1/M2/M3)

MPS (Metal Performance Shaders) is automatically used if available:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Post-Installation

### Run Tests

Ensure everything works correctly:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run with coverage report
pytest --cov=src tests/
```

**Expected result:** All tests should pass (green) or skip (yellow). Failures (red) indicate issues.

### Download Models (Optional)

Pre-download models to avoid delays during experiments:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download Qwen 3B (most commonly used)
print('Downloading Qwen 3B...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
print('✓ Download complete')
"
```

Models are cached in `~/.cache/huggingface/` by default.

### Verify API Service (Optional)

If you plan to use the API:

```bash
# Start server
uvicorn src.api_server:app --host 0.0.0.0 --port 8081 &

# Wait for startup
sleep 5

# Test health endpoint
curl http://localhost:8081/health

# Expected: {"status": "healthy"}

# Stop server
pkill -f "uvicorn src.api_server"
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure running from repository root
cd /path/to/nl2atl

# Ensure package installed
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=.
```

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Use smaller model (1-3B parameters)
2. Enable 4-bit quantization in `configs/models.yaml`:
   ```yaml
   load_in_4bit: true
   ```
3. Reduce batch size in `configs/experiments.yaml`:
   ```yaml
   training:
     batch_size: 4
   ```
4. Use gradient checkpointing (automatic for some models)

### PyTorch CUDA Mismatch

**Problem:** PyTorch installed but CUDA not detected

**Solution:**
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check system CUDA version
nvcc --version

# If mismatch, reinstall PyTorch with matching CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Azure Authentication Errors

**Problem:** `AuthenticationError: Invalid credentials`

**Solutions:**
1. Verify `.env` has correct `AZURE_API_KEY`
2. Check `AZURE_INFER_ENDPOINT` ends with `/`
3. Ensure deployment name matches `AZURE_INFER_MODEL`
4. Check firewall/VPN not blocking requests
5. Verify API key has not expired

### Hugging Face Token Issues

**Problem:** `Repository access denied` for gated models

**Solutions:**
1. Accept model license on Hugging Face website
2. Verify `HUGGINGFACE_TOKEN` in `.env`
3. Ensure token has "Read" permissions
4. Re-login: `huggingface-cli login`

### Slow Installation

**Problem:** `pip install` takes very long

**Solutions:**
1. Use faster mirror:
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
2. Install PyTorch first (largest package):
   ```bash
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

### Virtual Environment Not Activating

**Problem:** Command prompt doesn't show `(.venv)`

**Solutions:**

**Linux/macOS:**
```bash
# Ensure using correct activation script
source .venv/bin/activate

# Check shell
echo $SHELL
# If zsh: source .venv/bin/activate
# If bash: source .venv/bin/activate
```

**Windows:**
```powershell
# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.venv\Scripts\Activate.ps1
```

### Permission Denied Errors

**Problem:** `Permission denied` when installing

**Solutions:**
1. **Don't use sudo with pip** — this installs globally and breaks virtual environments
2. Ensure virtual environment activated
3. Check directory permissions:
   ```bash
   ls -la .venv
   # Should be owned by your user
   ```

---

## Verification Checklist

Before proceeding to experiments, verify:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NL2ATL installed (`pip install -e .`)
- [ ] CLI works (`nl2atl --help`)
- [ ] `.env` file configured (if using Azure/HF)
- [ ] Tests pass (`pytest`)
- [ ] GPU detected (optional, `torch.cuda.is_available()`)

---

## Next Steps

✅ **Installation complete!** Now you're ready to:

1. **Run your first experiment** → [Quick Start Guide](quickstart.md)
2. **Learn ATL syntax** → [ATL Primer](atl-primer.md)
3. **Explore CLI commands** → [Usage Guide](usage.md)
4. **Configure models** → [Configuration Guide](configuration.md)

---

## Getting Help

- **Documentation**: See [index.md](index.md) for full documentation map
- **Issues**: Report problems at [GitHub Issues](https://github.com/vladanaSTM/nl2atl/issues)
- **Community**: Check existing issues for solutions

---

## Uninstallation

To remove NL2ATL:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove cached models (optional)
rm -rf ~/.cache/huggingface/

# Remove cloned repository (optional)
cd ..
rm -rf nl2atl
```

---

**Ready to start?** → [Quick Start Guide](quickstart.md)
