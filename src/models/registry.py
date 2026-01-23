"""
Model loading, management, and generation utilities.
"""

import ctypes
import gc
import os
from typing import Optional, Tuple, Any

import torch
from ..infra.env import load_env
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from ..config import ModelConfig
from ..constants import Provider, ModelType
from ..infra.azure import AzureClient, AzureConfig

# Ensure environment variables are loaded
load_env()


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory."""
    # Multiple gc passes for circular references
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Force Python to release memory back to OS (Linux)
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

        gc.collect()
        torch.cuda.empty_cache()

        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total:.2f} GB total"
        )


def get_model_type(model_name: str) -> str:
    """Determine model type from name for prompt formatting."""
    model_lower = model_name.lower()

    if "azure" in model_lower or model_lower.startswith("gpt-"):
        return ModelType.GENERIC
    if "qwen" in model_lower:
        return ModelType.QWEN
    if "phi" in model_lower:
        return ModelType.PHI3
    if "mistral" in model_lower:
        return ModelType.MISTRAL
    if "llama" in model_lower:
        return ModelType.LLAMA
    if "gemma" in model_lower:
        return ModelType.GEMMA

    return ModelType.GENERIC


def is_large_model(model_name: str, cutoff_billion: int = 60) -> bool:
    """Detect models with >= cutoff_billion parameters from name tokens like 70B."""
    for token in model_name.replace("-", " ").replace("_", " ").split():
        t = token.lower()
        if t.endswith("b") and t[:-1].isdigit():
            try:
                if int(t[:-1]) >= cutoff_billion:
                    return True
            except ValueError:
                continue
    return False


def _build_bnb_config() -> BitsAndBytesConfig:
    """Build 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer with fallback handling."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
    except Exception as e:
        error_msg = str(e).lower()

        if "gated" in error_msg or "access" in error_msg:
            raise ValueError(
                f"Model {model_name} requires HuggingFace authentication."
            ) from e

        if any(x in error_msg for x in ("pretokenizer", "variant", "untagged enum")):
            print("Warning: Tokenizer deserialization issue. Attempting recovery...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    token=hf_token,
                )
                print("Recovered tokenizer with use_fast=False")
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False,
                )
        else:
            print(f"Warning: {e}. Trying without token...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def _load_hf_model(
    model_config: ModelConfig,
    for_training: bool,
    load_adapter: Optional[str],
) -> Tuple[Any, Any]:
    """Load a HuggingFace model and tokenizer."""
    clear_gpu_memory()
    print(f"Loading model: {model_config.name}")

    tokenizer = _load_tokenizer(model_config.name)

    use_4bit = model_config.load_in_4bit
    bnb_config = _build_bnb_config() if use_4bit else None
    load_attempted_4bit = use_4bit

    while True:
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # Special handling for very large models
        if torch.cuda.is_available() and is_large_model(
            model_config.name, cutoff_billion=60
        ):
            max_gpu_mem_gb = int(
                torch.cuda.get_device_properties(0).total_memory / (1024**3) - 2
            )
            model_kwargs["device_map"] = {"": "cuda:0"}
            model_kwargs["max_memory"] = {"cuda:0": f"{max_gpu_mem_gb}GiB"}

        if use_4bit:
            model_kwargs["quantization_config"] = bnb_config

        attn_impls = ["sdpa", "eager"]
        last_error = None

        for attn_impl in attn_impls:
            model_kwargs["attn_implementation"] = attn_impl
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.name,
                    **model_kwargs,
                )
                last_error = None
                break
            except (ValueError, NotImplementedError, ImportError, RuntimeError) as e:
                last_error = e
                error_msg = str(e).lower()

                if "out of memory" in error_msg and not load_attempted_4bit:
                    print("OOM loading model; retrying with 4-bit quantization.")
                    use_4bit = True
                    bnb_config = _build_bnb_config()
                    load_attempted_4bit = True
                    clear_gpu_memory()
                    break

                if attn_impl != attn_impls[-1] and "attention" in error_msg:
                    print(f"{attn_impl} not supported; trying next")
                    continue
                raise
        else:
            if load_attempted_4bit and last_error:
                raise last_error
            continue

        if last_error is None:
            break

    # Load adapter if specified
    if load_adapter:
        print(f"Loading adapter from: {load_adapter}")
        model = PeftModel.from_pretrained(model, load_adapter)

        # Only merge if NOT using 4-bit (merge corrupts weights with 4-bit)
        if not use_4bit:
            model = model.merge_and_unload()
            print("Adapter merged into base model")
        else:
            print("Keeping adapter separate (4-bit mode)")

    # Apply LoRA for training
    elif for_training:
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=model_config.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print(f"Model loaded. GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


def load_model(
    model_config: ModelConfig,
    for_training: bool = False,
    load_adapter: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer based on provider.

    Args:
        model_config: Model configuration
        for_training: Whether to prepare model for training
        load_adapter: Path to LoRA adapter to load

    Returns:
        Tuple of (model, tokenizer) - tokenizer is None for Azure models
    """
    if model_config.is_azure:
        azure_config = AzureConfig.from_env()
        api_model = (
            model_config.api_model
            or os.getenv("AZURE_INFER_MODEL")
            or model_config.name
        )
        client = AzureClient.from_config(azure_config, model=api_model)
        return client, None

    return _load_hf_model(model_config, for_training, load_adapter)


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate output from model."""
    # Azure models expose a simple generate() API
    if hasattr(model, "provider") and model.provider == "azure":
        return model.generate(prompt, max_new_tokens=max_new_tokens)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Phi-3 sometimes has cache issues; disable for safety
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    use_cache = model_type != "phi3"

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)
