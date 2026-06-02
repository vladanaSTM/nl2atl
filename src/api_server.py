"""FastAPI service for NL2ATL generation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import Config, ModelConfig
from .evaluation.exact_match import ExactMatchEvaluator
from .infra.env import load_env
from .models.few_shot import format_prompt
from .models.registry import generate, get_model_type, load_model
from .models.utils import resolve_model_key

load_env()

app = FastAPI(title="NL2ATL API", version="0.1.0")


class GenerateRequest(BaseModel):
    """Request schema for NL2ATL generation."""

    description: str = Field(..., min_length=1, description="Natural language input")
    model: Optional[str] = Field(
        default=None,
        description="Model key or name from configs/models.yaml (optional)",
    )
    few_shot: bool = Field(default=False, description="Enable few-shot prompting")
    num_few_shot: Optional[int] = Field(
        default=None, ge=0, description="Number of few-shot examples to include"
    )
    max_new_tokens: int = Field(
        default=128, ge=1, le=2048, description="Maximum new tokens to generate"
    )
    adapter: Optional[str] = Field(
        default=None,
        description=(
            "Optional LoRA adapter name or path. If relative, resolved against "
            "the models directory."
        ),
    )
    return_raw: bool = Field(
        default=False, description="Include raw model output in response"
    )

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        description = value.strip()
        if not description:
            raise ValueError("Description cannot be empty")
        return description

    @field_validator("model", "adapter")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class GenerateResponse(BaseModel):
    """Response schema for NL2ATL generation."""

    formula: str
    model_key: str
    model_name: str
    provider: str
    latency_ms: float
    raw_output: Optional[str] = None


@dataclass
class ModelEntry:
    """Loaded model resources kept by the API service."""

    model: Any
    tokenizer: Any
    model_config: ModelConfig
    model_type: str
    generation_lock: Lock = field(default_factory=Lock)


_MODEL_CACHE: Dict[str, ModelEntry] = {}
_MODEL_CACHE_LOCK = Lock()
_EVALUATOR = ExactMatchEvaluator()


@lru_cache(maxsize=1)
def _get_config() -> Config:
    models_config = os.getenv("NL2ATL_MODELS_CONFIG", "configs/models.yaml")
    experiments_config = os.getenv(
        "NL2ATL_EXPERIMENTS_CONFIG", "configs/experiments.yaml"
    )
    return Config.from_yaml(models_config, experiments_config)


def _get_default_model_key(config: Config) -> str:
    default_model = os.getenv("NL2ATL_DEFAULT_MODEL")
    if default_model:
        return resolve_model_key(default_model, config.models)

    if not config.models:
        raise HTTPException(status_code=500, detail="No models found in configuration")

    return next(iter(config.models.keys()))


def _resolve_request_model_key(config: Config, requested_model: Optional[str]) -> str:
    if requested_model:
        return resolve_model_key(requested_model, config.models)
    return _get_default_model_key(config)


def _build_model_entry(
    model_config: ModelConfig,
    model_type: str,
    *,
    load_adapter: Optional[str] = None,
) -> ModelEntry:
    model, tokenizer = load_model(
        model_config,
        for_training=False,
        load_adapter=load_adapter,
    )
    return ModelEntry(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        model_type=model_type,
    )


def _load_cached_model(model_key: str) -> ModelEntry:
    with _MODEL_CACHE_LOCK:
        if model_key in _MODEL_CACHE:
            return _MODEL_CACHE[model_key]

        config = _get_config()
        model_config = config.get_model(model_key)
        model_type = get_model_type(model_config.name)
        _MODEL_CACHE[model_key] = _build_model_entry(model_config, model_type)
        return _MODEL_CACHE[model_key]


def _resolve_adapter_path(adapter: str, config: Config) -> Path:
    candidate = Path(adapter).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config.models_dir).expanduser() / candidate
    if not candidate.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Adapter not found: {candidate}",
        )
    return candidate.resolve()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_formula(request: GenerateRequest) -> GenerateResponse:
    config = _get_config()

    try:
        model_key = _resolve_request_model_key(config, request.model)
        model_config = config.get_model(model_key)
    except KeyError as exc:
        status_code = 400 if request.model else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    model_type = get_model_type(model_config.name)

    adapter_path: Optional[str] = None
    if request.adapter:
        if model_config.is_azure:
            raise HTTPException(
                status_code=400,
                detail="Adapters are only supported for HuggingFace models",
            )
        adapter_path = str(_resolve_adapter_path(request.adapter, config))

    try:
        model_entry = (
            _build_model_entry(
                model_config,
                model_type,
                load_adapter=adapter_path,
            )
            if adapter_path
            else _load_cached_model(model_key)
        )
    except Exception as exc:
        failure = "Adapter load failed" if adapter_path else "Model load failed"
        raise HTTPException(status_code=500, detail=f"{failure}: {exc}") from exc

    num_few_shot = (
        request.num_few_shot
        if request.num_few_shot is not None
        else config.num_few_shot_examples
    )

    prompt = format_prompt(
        request.description,
        output_text=None,
        few_shot=request.few_shot,
        num_examples=num_few_shot,
        model_type=model_entry.model_type,
        tokenizer=model_entry.tokenizer,
    )

    start_time = time.perf_counter()
    try:
        with model_entry.generation_lock:
            raw_output = generate(
                model_entry.model,
                model_entry.tokenizer,
                prompt,
                max_new_tokens=request.max_new_tokens,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - start_time) * 1000

    formula = _EVALUATOR.clean_output(str(raw_output), model_entry.model_type)

    response = GenerateResponse(
        formula=formula,
        model_key=model_key,
        model_name=model_entry.model_config.name,
        provider=model_entry.model_config.provider,
        latency_ms=latency_ms,
        raw_output=str(raw_output) if request.return_raw else None,
    )

    return response
