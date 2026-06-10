"""LLM client wrappers for judge evaluation."""

from typing import Protocol, List

from ...infra.azure import AzureClient, AzureConfig


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
