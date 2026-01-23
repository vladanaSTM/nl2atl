"""LLM client wrappers for judge evaluation."""

from typing import Protocol, List, Any

from ...azure_utils import AzureClient, AzureConfig
from ...config import ModelConfig


def _get_model_registry():
    """Lazy import to avoid heavy dependencies during module import."""
    from ... import model_registry

    return model_registry


class JudgeClient(Protocol):
    """Protocol for LLM judge clients."""

    def complete(self, prompt: str, max_new_tokens: int = 256) -> str: ...

    def complete_batch(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> List[str]: ...


class AzureJudgeClient:
    """Azure OpenAI client wrapper for judge evaluation."""

    provider = "azure"

    def __init__(self, config: AzureConfig, model: str):
        self.client = AzureClient.from_config(config, model=model)

    def complete(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.client.generate(prompt, max_new_tokens=max_new_tokens)

    def complete_batch(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> List[str]:
        return [self.complete(p, max_new_tokens=max_new_tokens) for p in prompts]


class LocalJudgeClient:
    """Wrapper for local HuggingFace models as judges."""

    provider = "local"

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        mr = _get_model_registry()
        self.model, self.tokenizer = mr.load_model(model_config, for_training=False)

    def complete(self, prompt: str, max_new_tokens: int = 256) -> str:
        mr = _get_model_registry()
        raw = mr.generate(
            self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        if raw.startswith(prompt):
            return raw[len(prompt) :].strip()
        return raw.strip()

    def complete_batch(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> List[str]:
        return [self.complete(p, max_new_tokens=max_new_tokens) for p in prompts]


def get_client(provider: str, **kwargs: Any) -> JudgeClient:
    """Factory function to get appropriate client."""
    provider = provider.lower()
    if provider == "azure":
        config = kwargs.get("config")
        model = kwargs.get("model")
        if not isinstance(config, AzureConfig):
            raise ValueError("Azure client requires AzureConfig via 'config'.")
        if not model:
            raise ValueError("Azure client requires 'model'.")
        return AzureJudgeClient(config=config, model=model)
    if provider in {"huggingface", "local"}:
        model_config = kwargs.get("model_config")
        if not isinstance(model_config, ModelConfig):
            raise ValueError("Local client requires ModelConfig via 'model_config'.")
        return LocalJudgeClient(model_config=model_config)

    raise ValueError(f"Unknown provider: {provider}")
