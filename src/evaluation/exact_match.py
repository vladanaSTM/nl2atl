"""
Exact-match evaluator with output cleaning for ATL formula generation.
"""

import re
import time
from typing import Any, Dict, List

from .base import BaseEvaluator
from ..constants import TEMPORAL_OPERATORS
from ..data_utils import get_output_options, get_preferred_output
from ..models.few_shot import format_prompt


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator that normalizes model output and tracks exact-match accuracy."""

    def __init__(self):
        self.results: List[Dict] = []

    def _has_temporal_operator(self, text: str) -> bool:
        """Check if text contains any temporal operator."""
        return any(
            re.search(rf"(?<![a-zA-Z_]){op}(?![a-zA-Z_])", text)
            for op in TEMPORAL_OPERATORS
        )

    def _is_valid_formula(self, text: str) -> bool:
        """Check if text looks like a valid ATL formula."""
        return "<<" in text and ">>" in text and self._has_temporal_operator(text)

    def _extract_assistant_response(self, response: str, model_type: str) -> str:
        """Extract assistant response based on model type."""
        if model_type == "qwen":
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "")
            response = response.replace("<|im_start|>", "")
            response = response.replace("</s>", "")

        elif model_type == "mistral":
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1]
            response = response.replace("[INST]", "")
            response = response.replace("</s>", "")
            response = response.replace("<s>", "")

        elif model_type == "phi3":
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1]
            response = response.replace("<|end|>", "")

        else:
            for marker in ("Assistant:", "assistant:"):
                if marker in response:
                    response = response.rsplit(marker, 1)[-1]

        return response

    def _clean_think_tags(self, response: str) -> str:
        """Remove thinking/reasoning blocks."""
        # Handle closing </think> without opening tag
        if re.search(r"</think>", response, flags=re.IGNORECASE):
            response = re.split(r"</think>", response, flags=re.IGNORECASE)[-1]
        else:
            response = re.sub(
                r"<think>.*?</think>",
                "",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
        response = re.sub(r"</?think>", "", response, flags=re.IGNORECASE)
        return response

    def _strip_formatting_artifacts(self, response: str) -> str:
        """Remove common wrappers while preserving ATL syntax."""
        for token in (
            "<|im_end|>",
            "<|im_start|>",
            "<|end|>",
            "<|endoftext|>",
            "</s>",
            "<s>",
        ):
            response = response.replace(token, "")

        response = re.sub(r"```(?:atl|text|plaintext)?\s*", "", response, flags=re.I)
        response = response.replace("```", "")
        response = re.sub(
            r"^\s*(?:final\s+)?(?:formula|output|answer)\s*:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        return response

    def clean_output(self, response: str, model_type: str) -> str:
        """Extract generated formula from model response."""
        # Extract assistant response
        response = self._extract_assistant_response(response, model_type)

        # Clean thinking blocks
        response = self._clean_think_tags(response)

        # Remove wrappers and special tokens without damaging ATL operators
        response = self._strip_formatting_artifacts(response)
        response = re.sub(
            r"<[^>]*end[^>]*sentence[^>]*>", "", response, flags=re.IGNORECASE
        )

        # Clean up whitespace and quotes
        response = re.sub(r"\n\s*\n+", "\n", response)
        response = response.strip().strip('"').strip("'")

        # Try to extract formula using various methods
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # Deduplicate consecutive identical lines
        if lines:
            deduped = [lines[0]]
            for line in lines[1:]:
                if line != deduped[-1]:
                    deduped.append(line)
            lines = deduped

        # Method 1: Explicit <FORMULA> tags
        formula_tag = re.search(
            r"<formula>(.*?)</formula>", response, flags=re.DOTALL | re.IGNORECASE
        )
        if formula_tag and self._is_valid_formula(formula_tag.group(1)):
            return formula_tag.group(1).strip()

        # Method 2: First coalition fragment
        formula_match = re.search(r"<<[^>]+>>[^\.\r\n]*", response)
        if formula_match and self._is_valid_formula(formula_match.group(0)):
            return formula_match.group(0).strip()

        # Method 3: Line-based heuristics
        formula_like = [line for line in lines if self._is_valid_formula(line)]
        if not formula_like:
            formula_like = [line for line in lines if line.startswith("<<")]
        if not formula_like:
            formula_like = [line for line in lines if ">>" in line]

        if formula_like:
            response = formula_like[0]
        elif lines:
            response = lines[0]

        response = re.sub(r"(\\n)+$", "", response)
        return response.strip()

    def _normalize_symbols(self, formula: str) -> str:
        """Normalize Unicode/ASCII logical symbols to a common form."""
        if not formula:
            return formula

        # Unicode to ASCII
        normalized = (
            formula.replace("∧", "&&")
            .replace("∨", "||")
            .replace("¬", "!")
            .replace("→", "->")
            .replace("⇒", "->")
            .replace("↔", "<->")
            .replace("⇔", "<->")
        )

        # Word forms to symbols
        normalized = re.sub(r"\band\b", "&&", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bor\b", "||", normalized, flags=re.IGNORECASE)

        # Collapse repeated operators
        normalized = re.sub(r"\|\|+", "||", normalized)
        normalized = re.sub(r"&&+", "&&", normalized)

        return normalized

    def normalize(self, formula: str) -> str:
        """Normalize formula for comparison."""
        normalized = self._normalize_symbols(formula)
        return re.sub(r"\s+", "", normalized.strip().lower())

    def evaluate_single(
        self, prediction: Dict[str, Any], reference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single prediction-reference pair."""
        prediction_text = (
            prediction.get("generated")
            or prediction.get("prediction")
            or prediction.get("output")
            or prediction.get("text")
            or ""
        )
        reference_options = get_output_options(reference)
        reference_text = reference_options[0] if reference_options else ""
        exact_match = int(
            any(
                self.normalize(prediction_text) == self.normalize(reference_text)
                for reference_text in reference_options
            )
        )
        result = {
            "id": prediction.get("id") or reference.get("id"),
            "input": prediction.get("input") or reference.get("input"),
            "expected": reference_text,
            "expected_options": reference_options,
            "generated": prediction_text,
            "difficulty": prediction.get("difficulty") or reference.get("difficulty"),
            "exact_match": exact_match,
        }
        # Preserve latency if it was already computed
        if "latency_ms" in prediction:
            result["latency_ms"] = prediction["latency_ms"]
        return result

    def evaluate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate predictions.

        Supports both live model evaluation and precomputed
        prediction/reference evaluation.
        """
        if args and isinstance(args[0], list):
            predictions = args[0]
            references = args[1] if len(args) > 1 else kwargs.get("references")
            if references is None:
                raise ValueError(
                    "references must be provided when evaluating predictions"
                )
            return self.evaluate_predictions(predictions, references)

        return self.evaluate_model(*args, **kwargs)

    def evaluate_predictions(
        self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a list of prediction/reference pairs."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        self.results = []
        for prediction, reference in zip(predictions, references):
            self.results.append(self.evaluate_single(prediction, reference))

        return self.aggregate_metrics()

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        test_data: List[Dict],
        model_type: str,
        few_shot: bool = False,
        num_few_shot: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on test data.

        Args:
            model: Loaded model
            tokenizer: Tokenizer (None for Azure models)
            test_data: List of test items
            model_type: Model family type
            few_shot: Whether to use few-shot prompting
            num_few_shot: Number of few-shot examples
            verbose: Whether to print progress

        Returns:
            Dictionary of metrics
        """
        self.results = []

        if verbose:
            print(f"\nEvaluating {len(test_data)} examples...")
            print(f"Few-shot: {few_shot}, Model type: {model_type}")

        excluded_inputs = [item["input"] for item in test_data] if few_shot else None

        from ..models.registry import generate

        for i, item in enumerate(test_data):
            # Start timing
            start_time = time.perf_counter()

            prompt = format_prompt(
                input_text=item["input"],
                few_shot=few_shot,
                num_examples=num_few_shot,
                model_type=model_type,
                exclude_inputs=excluded_inputs,
                tokenizer=tokenizer,
            )

            response = generate(model, tokenizer, prompt)

            generated = self.clean_output(str(response), model_type)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            expected_options = get_output_options(item)
            exact_match = int(
                any(
                    self.normalize(expected) == self.normalize(generated)
                    for expected in expected_options
                )
            )

            # Build result dict
            result_dict = {
                "id": item.get("id", i),
                "input": item["input"],
                "expected": get_preferred_output(item),
                "expected_options": expected_options,
                "generated": generated,
                "difficulty": item.get("difficulty"),
                "exact_match": exact_match,
                "latency_ms": round(latency_ms, 2),
            }

            self.results.append(result_dict)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

        return self.aggregate_metrics()

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        n = len(self.results)
        if n == 0:
            return {"n_examples": 0, "exact_match": 0.0}

        exact_matches = sum(r["exact_match"] for r in self.results)

        metrics = {
            "n_examples": n,
            "exact_match": exact_matches / n,
        }

        return metrics
