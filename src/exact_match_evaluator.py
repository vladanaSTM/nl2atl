"""
Exact-match evaluator with output cleaning for ATL formula generation.
"""

import re
from typing import Dict, List, Any

from .model_registry import generate
from .few_shot import format_prompt


class ExactMatchEvaluator:
    """Evaluator that normalizes model output and tracks exact-match accuracy."""

    def __init__(self):
        self.results = []
        self.temporal_ops = ["G", "F", "X", "U", "W", "R"]

    def clean_output(self, response: str, model_type: str) -> str:
        """Extract generated formula from model response."""

        def has_temporal(text: str) -> bool:
            return any(
                re.search(rf"(?<![a-zA-Z_]){op}(?![a-zA-Z_])", text)
                for op in self.temporal_ops
            )

        def valid_formula(text: str) -> bool:
            return "<<" in text and ">>" in text and has_temporal(text)

        # Extract assistant response based on model type
        if model_type in ["qwen", "mistral"]:
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
            # Drop everything before the model's turn so the system prompt isn't treated as output
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1]
            elif "<start_of_turn>assistant" in response:
                response = response.split("<start_of_turn>assistant")[-1]
            response = response.replace("<start_of_turn>", "")
            response = response.replace("<end_of_turn>", "")
            response = response.replace("<eos>", "")
            response = response.replace("</s>", "")

        # Strip DeepSeek-style reasoning blocks and stray think tags.
        # If we see a closing </think> without an opening tag, treat everything after it as the final answer.
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

        # Remove end-of-sentence special markers that can leak into outputs
        response = re.sub(
            r"<[^>]*end[^>]*sentence[^>]*>", "", response, flags=re.IGNORECASE
        )

        response = response.replace("<|endoftext|>", "")
        # Collapse blank lines and trim quotes that sometimes wrap the formula
        response = re.sub(r"\n\s*\n+", "\n", response)
        response = response.strip().strip('"').strip("'")

        # Primary path: look for explicit <FORMULA> ... </FORMULA> tags (case-insensitive)
        formula_tag = re.search(
            r"<formula>(.*?)</formula>", response, flags=re.DOTALL | re.IGNORECASE
        )

        # Secondary path: take the first coalition fragment and stop before the first period/newline
        formula_match = re.search(r"<<[^>]+>>[^\.\r\n]*", response)

        # Then fall back to line-based heuristics and deduplication
        lines = [line.strip() for line in re.split(r"\r?\n", response) if line.strip()]
        if lines:
            deduped = []
            for line in lines:
                if not deduped or line != deduped[-1]:
                    deduped.append(line)
            lines = deduped

        if formula_tag and valid_formula(formula_tag.group(1)):
            response = formula_tag.group(1)
        elif formula_match and valid_formula(formula_match.group(0)):
            response = formula_match.group(0)
        else:
            formula_like = [line for line in lines if valid_formula(line)]
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

    def normalize(self, formula: str) -> str:
        normalized = self._normalize_symbols(formula)
        return re.sub(r"\s+", "", normalized.strip().lower())

    def _normalize_symbols(self, formula: str) -> str:
        """Normalize Unicode/ASCII logical symbols to a common form."""
        if not formula:
            return formula

        # Map common Unicode logical operators to ASCII equivalents
        normalized = (
            formula.replace("∧", "&&")
            .replace("∨", "||")
            .replace("¬", "!")
            .replace("→", "->")
            .replace("⇒", "->")
            .replace("↔", "<->")
            .replace("⇔", "<->")
        )

        # Normalize common ASCII variants to a canonical form
        normalized = re.sub(r"\band\b", "&&", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bor\b", "||", normalized, flags=re.IGNORECASE)

        # Collapse double variants (e.g., '|||' or '&&&') just in case
        normalized = re.sub(r"\|\|+", "||", normalized)
        normalized = re.sub(r"&&+", "&&", normalized)
        return normalized

    def evaluate(
        self,
        model,
        tokenizer,
        test_data: List[Dict],
        model_type: str,
        few_shot: bool = False,
        num_few_shot: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run evaluation on test data."""
        self.results = []

        if verbose:
            print(f"\nEvaluating {len(test_data)} examples...")
            print(f"Few-shot: {few_shot}, Model type: {model_type}")

        excluded_inputs = [item["input"] for item in test_data] if few_shot else None

        for i, item in enumerate(test_data):
            # Format prompt
            prompt = format_prompt(
                input_text=item["input"],
                few_shot=few_shot,
                num_examples=num_few_shot,
                model_type=model_type,
                exclude_inputs=excluded_inputs,
                tokenizer=tokenizer,
            )

            # Generate
            response = generate(model, tokenizer, prompt)
            generated = self.clean_output(response, model_type)

            # Score
            exact_match = int(
                self.normalize(item["output"]) == self.normalize(generated)
            )

            result = {
                "input": item["input"],
                "expected": item["output"],
                "generated": generated,
                "exact_match": exact_match,
            }
            self.results.append(result)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

        return self.aggregate_metrics()

    def aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        n = len(self.results)
        if n == 0:
            return {}

        exact_matches = [r["exact_match"] for r in self.results]
        exact_mean = sum(exact_matches) / n

        return {
            "n_examples": n,
            "exact_match": exact_mean,
        }
