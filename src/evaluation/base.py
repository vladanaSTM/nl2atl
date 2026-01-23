"""Base evaluator interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def evaluate(
        self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against references.

        Args:
            predictions: List of prediction records
            references: List of reference/ground-truth records

        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_single(
        self, prediction: Dict[str, Any], reference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single prediction-reference pair."""
        raise NotImplementedError
