"""
Exact-match evaluator with output cleaning for ATL formula generation.
"""

import re
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Iterable, Tuple

from .base import BaseEvaluator
from ..constants import TEMPORAL_OPERATORS
from ..models.registry import generate, GenerationResult
from ..models.few_shot import format_prompt


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator that normalizes model output and tracks exact-match accuracy."""

    def __init__(self):
        self.results: List[Dict] = []
        self.total_tokens_input: int = 0
        self.total_tokens_output: int = 0
        self.price_input_per_1k: Optional[float] = None
        self.price_output_per_1k: Optional[float] = None

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
        if model_type in ("qwen", "mistral"):
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "")
            response = response.replace("<|im_start|>", "")
            response = response.replace("</s>", "")

        elif model_type == "phi3":
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1]
            response = response.replace("<|end|>", "")

        elif model_type == "llama":
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[-1]
            response = response.replace("<|eot_id|>", "")

        elif model_type == "gemma":
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1]
            elif "<start_of_turn>assistant" in response:
                response = response.split("<start_of_turn>assistant")[-1]
            for tag in ("<start_of_turn>", "<end_of_turn>", "<eos>", "</s>"):
                response = response.replace(tag, "")

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

    def clean_output(self, response: str, model_type: str) -> str:
        """Extract generated formula from model response."""
        # Extract assistant response
        response = self._extract_assistant_response(response, model_type)

        # Clean thinking blocks
        response = self._clean_think_tags(response)

        # Remove special tokens
        response = re.sub(
            r"<[^>]*end[^>]*sentence[^>]*>", "", response, flags=re.IGNORECASE
        )
        response = response.replace("<|endoftext|>", "")

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
        reference_text = (
            reference.get("expected")
            or reference.get("output")
            or reference.get("gold")
            or reference.get("reference")
            or ""
        )
        exact_match = int(
            self.normalize(prediction_text) == self.normalize(reference_text)
        )
        result = {
            "id": prediction.get("id") or reference.get("id"),
            "input": prediction.get("input") or reference.get("input"),
            "expected": reference_text,
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

        Supports both legacy model-based evaluation and
        BaseEvaluator-style prediction/reference evaluation.
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
        price_input_per_1k: Optional[float] = None,
        price_output_per_1k: Optional[float] = None,
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
        # Reset token counters
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.price_input_per_1k = price_input_per_1k
        self.price_output_per_1k = price_output_per_1k

        # Determine if this is an Azure model (tokenizer is None)
        is_azure = tokenizer is None

        if verbose:
            print(f"\nEvaluating {len(test_data)} examples...")
            print(f"Few-shot: {few_shot}, Model type: {model_type}")

        excluded_inputs = [item["input"] for item in test_data] if few_shot else None

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

            # Get response with usage info
            result = generate(model, tokenizer, prompt, return_usage=True)

            tokens_estimated = False
            if isinstance(result, GenerationResult):
                response = result.text
                tokens_input = result.tokens_input
                tokens_output = result.tokens_output
                tokens_estimated = getattr(result, "usage_estimated", False)
                self.total_tokens_input += tokens_input
                self.total_tokens_output += tokens_output
            else:
                # Fallback if generate returns string (shouldn't happen with return_usage=True)
                response = result
                tokens_input = None
                tokens_output = None

            generated = self.clean_output(response, model_type)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            exact_match = int(
                self.normalize(item["output"]) == self.normalize(generated)
            )

            # Build result dict
            result_dict = {
                "id": item.get("id", i),
                "input": item["input"],
                "expected": item["output"],
                "generated": generated,
                "difficulty": item.get("difficulty"),
                "exact_match": exact_match,
                "latency_ms": round(latency_ms, 2),
            }

            # Add token counts
            if tokens_input is not None:
                result_dict["tokens_input"] = tokens_input
                result_dict["tokens_output"] = tokens_output
                result_dict["tokens_estimated"] = tokens_estimated

                if price_input_per_1k is not None and price_output_per_1k is not None:
                    price_in = Decimal(str(price_input_per_1k))
                    price_out = Decimal(str(price_output_per_1k))
                    tokens_in = Decimal(tokens_input)
                    tokens_out = Decimal(tokens_output)

                    cost_input = (tokens_in / Decimal("1000")) * price_in
                    cost_output = (tokens_out / Decimal("1000")) * price_out
                    cost_total = cost_input + cost_output

                    q = Decimal("0.000001")
                    result_dict["cost_input_usd"] = float(
                        cost_input.quantize(q, rounding=ROUND_HALF_UP)
                    )
                    result_dict["cost_output_usd"] = float(
                        cost_output.quantize(q, rounding=ROUND_HALF_UP)
                    )
                    result_dict["cost_usd"] = float(
                        cost_total.quantize(q, rounding=ROUND_HALF_UP)
                    )
                    result_dict["price_input_per_1k"] = price_input_per_1k
                    result_dict["price_output_per_1k"] = price_output_per_1k
                    result_dict["price_input_per_token"] = price_input_per_1k / 1000.0
                    result_dict["price_output_per_token"] = price_output_per_1k / 1000.0

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

        # Add token usage if tracked
        if self.total_tokens_input > 0:
            metrics["total_tokens_input"] = self.total_tokens_input
            metrics["total_tokens_output"] = self.total_tokens_output
            metrics["total_tokens"] = self.total_tokens_input + self.total_tokens_output

        total_cost = sum(Decimal(str(r.get("cost_usd", 0))) for r in self.results)
        total_cost_input = sum(
            Decimal(str(r.get("cost_input_usd", 0))) for r in self.results
        )
        total_cost_output = sum(
            Decimal(str(r.get("cost_output_usd", 0))) for r in self.results
        )
        if total_cost > 0:
            q = Decimal("0.000001")
            metrics["total_cost_usd"] = float(
                total_cost.quantize(q, rounding=ROUND_HALF_UP)
            )
            metrics["total_cost_input_usd"] = float(
                total_cost_input.quantize(q, rounding=ROUND_HALF_UP)
            )
            metrics["total_cost_output_usd"] = float(
                total_cost_output.quantize(q, rounding=ROUND_HALF_UP)
            )
            n_dec = Decimal(n)
            metrics["avg_cost_usd"] = float(
                (total_cost / n_dec).quantize(q, rounding=ROUND_HALF_UP)
            )
            metrics["avg_cost_input_usd"] = float(
                (total_cost_input / n_dec).quantize(q, rounding=ROUND_HALF_UP)
            )
            metrics["avg_cost_output_usd"] = float(
                (total_cost_output / n_dec).quantize(q, rounding=ROUND_HALF_UP)
            )

        if self.price_input_per_1k is not None and self.price_output_per_1k is not None:
            metrics["price_input_per_1k"] = self.price_input_per_1k
            metrics["price_output_per_1k"] = self.price_output_per_1k
            metrics["price_input_per_token"] = self.price_input_per_1k / 1000.0
            metrics["price_output_per_token"] = self.price_output_per_1k / 1000.0

        return metrics
