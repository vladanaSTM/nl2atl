"""Exact-match evaluator for ATL formula generation."""

from collections import Counter
import hashlib
import re
import time
from typing import Any, Dict, List

from .base import BaseEvaluator
from ..data_utils import get_output_options
from ..models.few_shot import (
    format_prompt,
    get_few_shot_example_id,
    get_few_shot_examples,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator that normalizes model output and tracks exact-match accuracy."""

    def __init__(self):
        self.results: List[Dict] = []

    def _extract_assistant_response(self, response: str, model_type: str) -> str:
        """Keep only the model answer bounded by chat start/stop tokens."""
        if model_type == "qwen":
            if "<|im_start|>assistant" in response:
                response = response.rsplit("<|im_start|>assistant", 1)[-1]
            response = self._truncate_at_first_stop(
                response, ("<|im_end|>", "<|endoftext|>")
            )

        elif model_type == "mistral":
            if "[/INST]" in response:
                response = response.rsplit("[/INST]", 1)[-1]
            response = self._truncate_at_first_stop(response, ("</s>",))

        elif model_type == "phi3":
            if "<|assistant|>" in response:
                response = response.rsplit("<|assistant|>", 1)[-1]
            response = self._truncate_at_first_stop(
                response, ("<|end|>", "<|endoftext|>")
            )

        else:
            for marker in ("Assistant:", "assistant:"):
                if marker in response:
                    response = response.rsplit(marker, 1)[-1]
            response = self._truncate_at_first_stop(
                response,
                ("<|im_end|>", "<|end|>", "<|endoftext|>", "</s>"),
            )

        return response

    def _truncate_at_first_stop(
        self, response: str, stop_tokens: tuple[str, ...]
    ) -> str:
        """Truncate at the first explicit model stop token, if present."""
        stops = [response.find(token) for token in stop_tokens if token in response]
        if stops:
            return response[: min(stops)]
        return response

    def clean_output(self, response: str, model_type: str) -> str:
        """Return the minimally cleaned model answer for evaluation."""
        response = self._extract_assistant_response(response, model_type)
        response = re.sub(r"\n\s*\n+", "\n", response)
        return response.strip().strip('"').strip("'")

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

    def split_formula_lines(self, text: str) -> List[str]:
        """Split model output into formula lines.

        Single-output predictions produce one line. QSA/multi-output predictions
        are expected to contain one formula per non-empty line.
        """
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]

    def outputs_exact_match(self, prediction_text: str, gold_outputs: List[str]) -> bool:
        """Return True iff predicted formulas match all gold outputs.

        The comparison is order-insensitive but multiplicity-sensitive. This means
        QSA outputs may be generated in either order, but producing only one
        required reading, duplicating a reading, or collapsing readings into one
        conjunctive formula does not count as exact match.
        """
        predicted_outputs = self.split_formula_lines(prediction_text)
        if not predicted_outputs or not gold_outputs:
            return False

        normalized_prediction = Counter(
            self.normalize(formula) for formula in predicted_outputs
        )
        normalized_gold = Counter(self.normalize(formula) for formula in gold_outputs)
        return normalized_prediction == normalized_gold

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
        reference_text = "\n".join(reference_options)
        exact_match = int(self.outputs_exact_match(prediction_text, reference_options))
        result = {
            "id": prediction.get("id") or reference.get("id"),
            "input": prediction.get("input") or reference.get("input"),
            "expected": reference_text,
            "expected_options": reference_options,
            "expected_outputs": reference_options,
            "generated": prediction_text,
            "predicted_outputs": self.split_formula_lines(prediction_text),
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
            few_shot_seed = 42

            prompt = format_prompt(
                input_text=item["input"],
                few_shot=few_shot,
                num_examples=num_few_shot,
                model_type=model_type,
                exclude_inputs=excluded_inputs,
                tokenizer=tokenizer,
            )

            few_shot_example_ids = (
                [
                    get_few_shot_example_id(example)
                    for example in get_few_shot_examples(
                        n=num_few_shot,
                        seed=few_shot_seed,
                        exclude_inputs=excluded_inputs,
                    )
                ]
                if few_shot
                else []
            )

            prompt_sha256 = _sha256_text(prompt)

            try:
                response = generate(model, tokenizer, prompt, return_usage=True)
            except TypeError:
                response = generate(model, tokenizer, prompt)

            raw_generation = getattr(response, "text", response)
            usage = getattr(response, "usage", None)
            usage_estimated = bool(getattr(response, "usage_estimated", False))
            generated = self.clean_output(str(raw_generation), model_type)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            expected_options = get_output_options(item)
            exact_match = int(self.outputs_exact_match(generated, expected_options))

            # Build result dict
            result_dict = {
                "id": item.get("id", i),
                "input": item["input"],
                "expected": "\n".join(expected_options),
                "expected_options": expected_options,
                "expected_outputs": expected_options,
                "generated": generated,
                "predicted_outputs": self.split_formula_lines(generated),
                "raw_generation": str(raw_generation),
                "exact_match": exact_match,
                "latency_ms": round(latency_ms, 2),
                "generation_prompt_sha256": prompt_sha256,
                "generation_config": {
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "temperature": 0,
                    "model_type": model_type,
                    "few_shot": few_shot,
                    "num_few_shot": num_few_shot if few_shot else 0,
                    "few_shot_seed": few_shot_seed if few_shot else None,
                },
                "few_shot_example_ids": few_shot_example_ids,
                "token_usage": usage,
                "usage_estimated": usage_estimated,
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
