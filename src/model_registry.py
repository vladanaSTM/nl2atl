"""
Model loading and management.
"""

import os
import gc
import json
import time
import requests
import torch
from typing import Tuple, Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .config import ModelConfig

# Ensure environment variables from .env are loaded for Azure access.
load_dotenv()

# Silence noisy warnings when verify_ssl is intentionally disabled.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


class AzureClient:
    """Simple client for the Azure OpenAI proxy."""

    provider = "azure"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        use_cache: bool = True,
        api_version: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.use_cache = use_cache
        self.api_version = api_version
        self.verify_ssl = verify_ssl if verify_ssl is not None else False

        # Session with retries to survive transient disconnects from proxy.
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _build_headers(self) -> dict:
        headers = {
            # Azure expects api-key header, not Authorization bearer.
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Connection": "close",  # avoid keep-alive issues with the proxy
        }
        return headers

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # If caller provides the fully qualified chat/completions URL, use it.
        if self.endpoint.endswith("/chat/completions"):
            url = self.endpoint
        else:
            # Default Azure-compatible path; append api-version as query
            url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions"
            api_ver = self.api_version or "2024-08-01-preview"
            url = f"{url}?api-version={api_ver}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": False,
        }

        last_error = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=(10, 120),
                    verify=self.verify_ssl,
                )
                if response.status_code >= 400:
                    # Include body for easier troubleshooting
                    body = response.text[:500]
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                response.raise_for_status()
                break
            except Exception as e:
                last_error = e
                if attempt == 2:
                    raise RuntimeError(
                        f"Azure request failed after retries: {e}"
                    ) from e
                time.sleep(1.5 * (attempt + 1))

        try:
            data = response.json()
        except json.JSONDecodeError:
            return response.text

        # OpenAI-style response parsing
        choices = data.get("choices")
        if choices:
            first = choices[0]
            # Prefer chat-style content
            message = first.get("message") or {}
            content = message.get("content")
            if content:
                return content
            # Fallback to completion-style text
            if "text" in first and first["text"]:
                return first["text"]

        # If schema differs, return best-effort string
        return json.dumps(data)


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
    if "azure" in model_lower or model_lower.startswith("gpt-"):
        return "generic"
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
    # Route to Azure backend when requested; no GPU allocations.
    if model_config.provider.lower() == "azure":
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_INFER_ENDPOINT")
        api_version = os.getenv("AZURE_API_VERSION")
        use_cache = os.getenv("AZURE_USE_CACHE", "true").lower() == "true"
        verify_ssl_env = os.getenv("AZURE_VERIFY_SSL", "false").lower()
        verify_ssl = verify_ssl_env in ["1", "true", "yes"]
        api_model = (
            model_config.api_model
            or os.getenv("AZURE_INFER_MODEL")
            or model_config.name
        )

        if not api_key:
            raise ValueError(
                "AZURE_API_KEY is not set; populate .env or environment variables."
            )
        if not endpoint:
            raise ValueError(
                "AZURE_INFER_ENDPOINT is not set; populate .env or environment variables."
            )

        client = AzureClient(
            endpoint=endpoint,
            api_key=api_key,
            model=api_model,
            use_cache=use_cache,
            api_version=api_version,
            verify_ssl=verify_ssl,
        )

        # Tokenizer is not required for remote inference.
        return client, None

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

        if torch.cuda.is_available() and is_large_model(
            model_config.name, cutoff_billion=60
        ):
            # Avoid CPU/disk offload heuristics for 60B+ models; keep them on cuda:0.
            max_gpu_mem_gb = int(
                torch.cuda.get_device_properties(0).total_memory / (1024**3) - 2
            )
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
    # Remote Azure models expose a simple generate() API.
    if hasattr(model, "provider") and getattr(model, "provider") == "azure":
        return model.generate(prompt, max_new_tokens=max_new_tokens)

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
