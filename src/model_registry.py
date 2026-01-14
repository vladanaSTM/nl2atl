"""
Model loading and management.
"""

import os
import torch
import gc
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .config import ModelConfig


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f} / {total:.2f} GB")


def get_model_type(model_name: str) -> str:
    """Determine model type from name for prompt formatting."""
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return "qwen"
    elif "phi" in model_lower:
        return "phi3"
    elif "mistral" in model_lower:
        return "mistral"
    elif "llama" in model_lower:
        return "llama"
    elif "gemma" in model_lower:
        return "gemma"
    else:
        return "generic"


def is_large_model(model_name: str, cutoff_billion: int = 60) -> bool:
    """Heuristic: detect models with >= cutoff_billion parameters from name tokens like 70B."""
    for token in model_name.replace("-", " ").replace("_", " ").split():
        t = token.lower()
        if t.endswith("b") and t[:-1].isdigit():
            try:
                if int(t[:-1]) >= cutoff_billion:
                    return True
            except ValueError:
                continue
    return False


def load_model(
    model_config: ModelConfig,
    for_training: bool = False,
    load_adapter: Optional[str] = None,
) -> Tuple[any, any]:
    """
    Load model and tokenizer.

    Args:
        model_config: Model configuration
        for_training: Whether to prepare for training (apply LoRA)
        load_adapter: Path to load fine-tuned adapter from
    """
    clear_gpu_memory()

    print(f"Loading model: {model_config.name}")

    def build_bnb_config() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    use_4bit = model_config.load_in_4bit
    bnb_config = build_bnb_config() if use_4bit else None

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
    except Exception as e:
        error_msg = str(e).lower()
        # Check for authentication errors
        if "gated" in error_msg or "access" in error_msg:
            raise ValueError(
                f"Model {model_config.name} requires HuggingFace authentication. "
                f"Please set HUGGINGFACE_TOKEN or run: huggingface-cli login"
            )
        # For tokenizer deserialization errors (e.g., Mistral), retry with legacy tokenizer
        if (
            "pretokenizer" in error_msg
            or "variant" in error_msg
            or "untagged enum" in error_msg
        ):
            print(f"Warning: Tokenizer deserialization issue. Attempting recovery...")
            try:
                # Try loading with use_fast=False for Mistral and similar issues
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.name,
                    trust_remote_code=True,
                    use_fast=False,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                )
                print(f"Recovered tokenizer with use_fast=False")
            except Exception as retry_error:
                print(f"Warning: Tokenizer recovery failed. Trying without token...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.name, trust_remote_code=True, use_fast=False
                )
        else:
            # For other errors, try without token as fallback
            print(f"Warning: {e}. Trying without token...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.name, trust_remote_code=True
            )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_attempted_4bit = use_4bit
    last_error = None
    while True:
        # Default to auto placement; override to single-GPU placement only for very large models.
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "device_map": "auto",
        }

        if torch.cuda.is_available() and is_large_model(model_config.name, cutoff_billion=60):
            # Avoid CPU/disk offload heuristics for 60B+ models; keep them on cuda:0.
            max_gpu_mem_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024**3) - 2)
            model_kwargs["device_map"] = {"": "cuda:0"}
            model_kwargs["max_memory"] = {"cuda:0": f"{max_gpu_mem_gb}GiB"}

        if use_4bit:
            model_kwargs["quantization_config"] = bnb_config


        # Avoid flash attention per user request: prefer SDPA, then eager
        attn_impls = ["sdpa", "eager"]

        for attn_impl in attn_impls:
            model_kwargs["attn_implementation"] = attn_impl
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.name, **model_kwargs
                )
                last_error = None
                break
            except (ValueError, NotImplementedError, ImportError, RuntimeError) as e:
                last_error = e
                error_msg = str(e).lower()
                # If the failure is OOM and we have not tried 4-bit yet, retry with 4-bit
                if "out of memory" in error_msg and not load_attempted_4bit:
                    print(
                        "OOM loading model in full precision; retrying with 4-bit quantization."
                    )
                    use_4bit = True
                    bnb_config = build_bnb_config()
                    load_attempted_4bit = True
                    clear_gpu_memory()
                    break
                # If attention implementation unsupported, try next option
                if attn_impl != attn_impls[-1] and "attention" in error_msg:
                    print(
                        f"{attn_impl} not supported ({e}); trying next attention implementation"
                    )
                    continue
                raise
        else:
            # No break occurred; if we exited due to OOM retry we loop again
            if load_attempted_4bit and last_error:
                raise last_error
            continue
        # Break outer while when model successfully loaded
        if last_error is None:
            break

    # Load adapter if specified
    if load_adapter:
        print(f"Loading adapter from: {load_adapter}")
        model = PeftModel.from_pretrained(model, load_adapter)
        model = model.merge_and_unload()

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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate output from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Newer Phi-3 builds sometimes return DynamicCache without seen_tokens; disable cache to avoid it
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    use_cache = False if model_type == "phi3" else True

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

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response
