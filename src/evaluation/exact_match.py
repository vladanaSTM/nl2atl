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

    def _has_balanced_delimiters(self, text: str) -> bool:
        """Check structural delimiters that commonly break in run-on outputs."""
        if text.count("<<") != text.count(">>"):
            return False
        paren_depth = 0
        for char in text:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
                if paren_depth < 0:
                    return False
        return paren_depth == 0

    def _is_degenerate_repetition(self, text: str) -> bool:
        """Detect repeated-token collapses such as Phi's whowho/cancer loops."""
        stripped = re.sub(r"\s+", " ", text.strip().lower())
        if not stripped:
            return False

        repeated_word = re.search(
            r"\b([a-z][a-z0-9_-]{1,24})\b(?:\s+\1\b){7,}", stripped
        )
        if repeated_word:
            return True

        compact = re.sub(r"[^a-z0-9_<>!&|()\[\],-]+", "", stripped)
        if len(compact) < 24:
            return False
        for width in range(2, min(16, len(compact) // 8) + 1):
            unit = compact[:width]
            if not unit.strip("0123456789"):
                continue
            repeated = unit * (len(compact) // width)
            remainder = compact[len(repeated) :]
            if compact.startswith(repeated) and len(repeated) >= len(compact) * 0.9:
                return True
            if remainder and (repeated + unit).startswith(compact):
                return True
        return False

    def _is_valid_formula(self, text: str) -> bool:
        """Check if text looks like a valid ATL formula."""
        text = text.strip()
        return (
            "<<" in text
            and ">>" in text
            and self._has_temporal_operator(text)
            and self._has_balanced_delimiters(text)
            and not text.rstrip().endswith((",", "&&", "||", "->", "<->"))
            and not self._is_degenerate_repetition(text)
        )

    def _extract_first_formula(self, response: str) -> str:
        """Extract the first ATL formula while dropping explanations/candidates."""
        coalition_start = response.find("<<")
        if coalition_start < 0:
            return ""

        formula_start = coalition_start
        prefix_index = coalition_start - 1
        while prefix_index >= 0 and response[prefix_index].isspace():
            prefix_index -= 1
        if prefix_index >= 0 and response[prefix_index] == "!":
            formula_start = prefix_index

        explanation_start = re.compile(
            r"\s*(?:[,;]\s*)?(?:"
            r"where\b|which\b|meaning\b|that\s+is\b|"
            r"i\.e\.|e\.g\.|or\s+equivalently\b|"
            r"and\s+equivalently\b|equivalently\b|represents\b|denotes\b"
            r")",
            flags=re.IGNORECASE,
        )
        next_candidate_start = re.compile(
            r"\s*(?:\band\b|\bor\b)\s*(?=!?.*?<<)", flags=re.IGNORECASE
        )

        end = len(response)
        paren_depth = 0
        in_coalition = False
        index = formula_start

        while index < len(response):
            if response.startswith("<<", index):
                in_coalition = True
                index += 2
                continue

            if in_coalition:
                if response.startswith(">>", index):
                    in_coalition = False
                    index += 2
                    continue
                index += 1
                continue

            char = response[index]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)

            if paren_depth == 0:
                remainder = response[index:]
                if char in "\r\n.,;":
                    end = index
                    break
                if explanation_start.match(remainder):
                    end = index
                    break
                if next_candidate_start.match(remainder):
                    end = index
                    break

            index += 1

        if in_coalition or paren_depth != 0:
            return ""

        return response[formula_start:end].strip().rstrip(",;:")

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
        if formula_tag:
            tagged_formula = self._extract_first_formula(formula_tag.group(1))
            if self._is_valid_formula(tagged_formula):
                return tagged_formula

        # Method 2: First complete ATL formula before explanations/alternatives
        first_formula = self._extract_first_formula(response)
        if self._is_valid_formula(first_formula):
            return first_formula

        # Method 3: Line-based heuristics
        formula_like = [line for line in lines if self._is_valid_formula(line)]

        if formula_like:
            return formula_like[0].strip()

        return ""

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
        invalid_outputs = sum(1 for r in self.results if not r.get("generated"))

        metrics = {
            "n_examples": n,
            "exact_match": exact_matches / n,
            "invalid_outputs": invalid_outputs,
            "invalid_output_rate": invalid_outputs / n,
        }

        return metrics
