"""Evaluation subpackage."""

from .base import BaseEvaluator
from .difficulty import DifficultyClassifier
from .judge_agreement import generate_agreement_report

__all__ = [
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "DifficultyClassifier",
    "generate_agreement_report",
    "LLMJudgeEvaluator",
    "run_llm_judge",
    "build_efficiency_report",
]


def __getattr__(name: str):
    if name == "ExactMatchEvaluator":
        from .exact_match import ExactMatchEvaluator

        return ExactMatchEvaluator
    if name in {"LLMJudgeEvaluator", "run_llm_judge"}:
        from .llm_judge import LLMJudgeEvaluator, run_llm_judge

        return {"LLMJudgeEvaluator": LLMJudgeEvaluator, "run_llm_judge": run_llm_judge}[
            name
        ]
    if name == "build_efficiency_report":
        from .model_efficiency import build_efficiency_report

        return build_efficiency_report
    raise AttributeError(f"module {__name__} has no attribute {name}")
