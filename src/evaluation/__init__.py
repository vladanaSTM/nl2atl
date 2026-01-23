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
    raise AttributeError(f"module {__name__} has no attribute {name}")
