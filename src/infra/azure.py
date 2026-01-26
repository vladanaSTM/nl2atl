"""Azure OpenAI client and configuration utilities."""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any  # <-- ADD Union, Dict, Any

import requests
from .env import load_env
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

# Load environment variables
load_env()

# Silence warnings when SSL verification is intentionally disabled
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


# ============== Add GenerationResult dataclass ==============
@dataclass
class GenerationResult:
    """Result from model generation including usage stats."""

    text: str
    usage: Optional[Dict[str, int]] = None

    @property
    def tokens_input(self) -> int:
        return self.usage.get("tokens_input", 0) if self.usage else 0

    @property
    def tokens_output(self) -> int:
        return self.usage.get("tokens_output", 0) if self.usage else 0

    @property
    def tokens_total(self) -> int:
        return self.usage.get("tokens_total", 0) if self.usage else 0


@dataclass
class AzureConfig:
    """Azure API configuration loaded from environment variables."""

    api_key: str
    endpoint: str
    api_version: Optional[str] = None
    api_model: Optional[str] = None
    use_cache: bool = True
    verify_ssl: bool = False

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Load Azure configuration from environment variables."""
        load_env()
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_INFER_ENDPOINT")
        api_version = os.getenv("AZURE_API_VERSION")
        api_model = os.getenv("AZURE_INFER_MODEL")
        use_cache = os.getenv("AZURE_USE_CACHE", "true").lower() == "true"
        verify_ssl_env = os.getenv("AZURE_VERIFY_SSL", "false").lower()
        verify_ssl = verify_ssl_env in ("1", "true", "yes")

        if not api_key:
            raise ValueError("AZURE_API_KEY environment variable is not set")
        if not endpoint:
            raise ValueError("AZURE_INFER_ENDPOINT environment variable is not set")

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            api_model=api_model,
            use_cache=use_cache,
            verify_ssl=verify_ssl,
        )

    @classmethod
    def from_env_optional(cls) -> Optional["AzureConfig"]:
        """Load Azure configuration, returning None if not configured."""
        try:
            return cls.from_env()
        except ValueError:
            return None


class AzureClient:
    """Client for Azure OpenAI API."""

    provider = "azure"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        use_cache: bool = True,
        api_version: Optional[str] = None,
        verify_ssl: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.use_cache = use_cache
        self.api_version = api_version or "2024-08-01-preview"
        self.verify_ssl = verify_ssl

        # Session with retries for transient failures
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

    @classmethod
    def from_config(
        cls,
        config: AzureConfig,
        model: str,
    ) -> "AzureClient":
        """Create client from AzureConfig."""
        return cls(
            endpoint=config.endpoint,
            api_key=config.api_key,
            model=model,
            use_cache=config.use_cache,
            api_version=config.api_version,
            verify_ssl=config.verify_ssl,
        )

    def _build_headers(self) -> dict:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Connection": "close",
        }

    def _build_url(self) -> str:
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        base_url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions"
        return f"{base_url}?api-version={self.api_version}"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        return_usage: bool = False,  # <-- NEW parameter
    ) -> Union[str, GenerationResult]:  # <-- UPDATE return type
        """
        Generate a response from the Azure OpenAI API.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            return_usage: If True, return GenerationResult with token counts

        Returns:
            Generated text string, or GenerationResult if return_usage=True
        """
        url = self._build_url()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": False,
        }

        response = None
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
                    body = response.text[:500]
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(
                        f"Azure request failed after retries: {e}"
                    ) from e
                time.sleep(1.5 * (attempt + 1))

        try:
            data = response.json()
        except json.JSONDecodeError:
            if return_usage:
                return GenerationResult(text=response.text, usage=None)
            return response.text

        # ============== Extract usage information ==============
        usage = None
        usage_data = data.get("usage")
        if isinstance(usage_data, dict):
            usage = {
                "tokens_input": usage_data.get("prompt_tokens", 0),
                "tokens_output": usage_data.get("completion_tokens", 0),
                "tokens_total": usage_data.get("total_tokens", 0),
            }

        # Parse OpenAI-style response
        content = None
        choices = data.get("choices", [])
        if choices:
            first = choices[0]
            message = first.get("message", {})
            content = message.get("content")
            if not content and "text" in first and first["text"]:
                content = first["text"]

        if content is None:
            content = json.dumps(data)

        # ============== Return based on return_usage flag ==============
        if return_usage:
            return GenerationResult(text=content, usage=usage)
        return content
