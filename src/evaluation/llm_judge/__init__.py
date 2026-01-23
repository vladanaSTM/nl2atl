"""LLM-as-a-judge evaluation subpackage."""

from .pipeline import (
    LLMJudge,
    LLMJudgeEvaluator,
    JudgeDecision,
    normalize_text,
    extract_prediction_items,
    evaluate_prediction_file,
    build_summary,
    build_summary_notebook,
    run_llm_judge,
)
from .client import JudgeClient, AzureJudgeClient, LocalJudgeClient, get_client
from .prompts import PROMPT_VERSION, JudgePromptConfig, format_judge_prompt
from .parser import JudgeVerdict, parse_judge_response
from .metrics import (
    JudgeMetrics,
    compute_metrics,
    compute_metrics_with_difficulty,
    _empty_metrics,
)

__all__ = [
    "LLMJudge",
    "LLMJudgeEvaluator",
    "JudgeDecision",
    "JudgeClient",
    "AzureJudgeClient",
    "LocalJudgeClient",
    "get_client",
    "JudgePromptConfig",
    "format_judge_prompt",
    "PROMPT_VERSION",
    "JudgeVerdict",
    "parse_judge_response",
    "JudgeMetrics",
    "compute_metrics",
    "compute_metrics_with_difficulty",
    "_empty_metrics",
    "normalize_text",
    "extract_prediction_items",
    "evaluate_prediction_file",
    "build_summary",
    "build_summary_notebook",
    "run_llm_judge",
]
