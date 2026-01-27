"""FastAPI service for NL2ATL generation."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
        default=None, description="Number of few-shot examples to include"
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


class GenerateResponse(BaseModel):
    """Response schema for NL2ATL generation."""

    formula: str
    model_key: str
    model_name: str
    provider: str
    latency_ms: float
    raw_output: Optional[str] = None


_MODEL_CACHE: Dict[str, Tuple[Any, Any, ModelConfig, str]] = {}
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


def _load_cached_model(model_key: str) -> Tuple[Any, Any, ModelConfig, str]:
    config = _get_config()
    resolved_key = resolve_model_key(model_key, config.models)

    if resolved_key in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_key]

    model_config = config.get_model(resolved_key)
    model, tokenizer = load_model(model_config, for_training=False)
    model_type = get_model_type(model_config.name)

    _MODEL_CACHE[resolved_key] = (model, tokenizer, model_config, model_type)
    return _MODEL_CACHE[resolved_key]


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_formula(request: GenerateRequest) -> GenerateResponse:
    description = request.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    config = _get_config()
    model_key = request.model or _get_default_model_key(config)

    adapter_path: Optional[str] = None
    if request.adapter:
        models_dir = Path(config.models_dir)
        candidate = Path(request.adapter)
        if not candidate.is_absolute():
            candidate = models_dir / candidate
        if not candidate.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Adapter not found: {candidate}",
            )
        adapter_path = str(candidate)

    try:
        model, tokenizer, model_config, model_type = _load_cached_model(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Model load failed: {exc}"
        ) from exc

    if adapter_path:
        if model_config.is_azure:
            raise HTTPException(
                status_code=400,
                detail="Adapters are only supported for HuggingFace models",
            )
        try:
            model, tokenizer = load_model(
                model_config, for_training=False, load_adapter=adapter_path
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Adapter load failed: {exc}"
            ) from exc

    num_few_shot = request.num_few_shot or config.num_few_shot_examples

    prompt = format_prompt(
        description,
        output_text=None,
        few_shot=request.few_shot,
        num_examples=num_few_shot,
        model_type=model_type,
        tokenizer=tokenizer,
    )

    start_time = time.perf_counter()
    try:
        raw_output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - start_time) * 1000

    formula = _EVALUATOR.clean_output(str(raw_output), model_type)

    response = GenerateResponse(
        formula=formula,
        model_key=model_key,
        model_name=model_config.name,
        provider=model_config.provider,
        latency_ms=latency_ms,
        raw_output=str(raw_output) if request.return_raw else None,
    )

    return response
