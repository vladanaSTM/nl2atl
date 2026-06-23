"""LLM client wrappers for judge evaluation."""

from typing import Protocol, List, TYPE_CHECKING

from ...infra.azure import AzureClient, AzureConfig

if TYPE_CHECKING:
    from ...config import ModelConfig


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


class HFJudgeClient:
    """Local, self-hosted HuggingFace judge client (e.g. run via SLURM).

    The model is loaded once and reused across ``complete`` calls. Heavy imports
    (torch/transformers) are deferred to construction so the judge package stays
    importable without a GPU stack present.
    """

    provider = "huggingface"

    def __init__(self, model_config: "ModelConfig"):
        from ...models.registry import load_model  # deferred heavy import
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Local judge '{model_config.short_name}' needs a CUDA GPU, but none "
                "is visible. Request one on the cluster (e.g. --gres=gpu:1, or use "
                "`nl2atl llm-judge --slurm`); 4-bit judges cannot run on CPU."
            )

        self.model_config = model_config
        self.model, self.tokenizer = load_model(model_config)

    def complete(self, prompt: str, max_new_tokens: int = 256) -> str:
        from ...models.registry import generate  # deferred heavy import

        text = self._apply_chat_template(prompt)
        return generate(self.model, self.tokenizer, text, max_new_tokens=max_new_tokens)

    def complete_batch(
        self, prompts: List[str], max_new_tokens: int = 256
    ) -> List[str]:
        return [self.complete(p, max_new_tokens=max_new_tokens) for p in prompts]

    def _apply_chat_template(self, prompt: str) -> str:
        """Present the judge prompt as a chat turn when the tokenizer supports it.

        Off-the-shelf instruct judges (Llama-3.3, Gemma-2) follow the JSON rubric
        far more reliably as a chat message than as raw text. Falls back to the
        raw prompt when no chat template is defined (e.g. a base model).
        """
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if apply is None or getattr(self.tokenizer, "chat_template", None) is None:
            return prompt
        try:
            return apply(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Heterogeneous third-party templates may reject a bare user turn;
            # fall back to the raw prompt rather than failing the whole run.
            return prompt
