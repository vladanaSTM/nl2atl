"""Evaluation subpackage."""

from .base import BaseEvaluator
from .difficulty import DifficultyClassifier
from .judge_agreement import generate_agreement_report

__all__ = [
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "DifficultyClassifier",
    "generate_agreement_report",
]


def __getattr__(name: str):
    if name == "ExactMatchEvaluator":
        from .exact_match import ExactMatchEvaluator

        return ExactMatchEvaluator
    raise AttributeError(f"module {__name__} has no attribute {name}")
