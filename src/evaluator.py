"""
Evaluation metrics and framework.
"""

import re
import numpy as np
from typing import Dict, List, Any

from .model_registry import generate
from .few_shot import format_prompt


class ATLEvaluator:
    """Evaluator for ATL formula generation."""

    def __init__(self):
        self.results = []
        self.agent_pattern = r"<<([^>]+)>>"
        self.temporal_ops = ["G", "F", "X", "U", "W", "R"]
        self.logical_ops = ["->", "&", "|", "!"]

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

        return response.strip()

    def normalize(self, formula: str) -> str:
        return re.sub(r"\s+", "", formula.strip().lower())

    def extract_agents(self, formula: str) -> List[str]:
        matches = re.findall(self.agent_pattern, formula)
        agents = []
        for match in matches:
            agents.extend([a.strip() for a in match.split(",")])
        return agents

    def extract_temporal_operators(self, formula: str) -> List[str]:
        found = []
        for op in self.temporal_ops:
            pattern = rf"(?<![a-zA-Z_]){op}(?![a-zA-Z_])"
            if re.search(pattern, formula):
                found.append(op)
        return found

    def extract_logical_operators(self, formula: str) -> List[str]:
        return [op for op in self.logical_ops if op in formula]

    def check_syntax(self, formula: str) -> Dict[str, bool]:
        checks = {
            "has_agent_brackets": bool(re.search(self.agent_pattern, formula)),
            "brackets_balanced": formula.count("(") == formula.count(")"),
            "has_temporal_op": any(
                re.search(rf"(?<![a-zA-Z_]){op}(?![a-zA-Z_])", formula)
                for op in self.temporal_ops
            ),
            "agent_brackets_closed": formula.count("<<") == formula.count(">>"),
        }
        checks["all_valid"] = all(checks.values())
        return checks

    def compute_f1(self, expected_set: set, generated_set: set) -> float:
        if not expected_set:
            return 1.0
        if not generated_set:
            return 0.0
        precision = len(expected_set & generated_set) / len(generated_set)
        recall = len(expected_set & generated_set) / len(expected_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_scores(self, expected: str, generated: str) -> Dict[str, float]:
        scores = {}

        # Exact match
        scores["exact_match"] = float(
            self.normalize(expected) == self.normalize(generated)
        )

        # Agent F1
        exp_agents = set(a.lower() for a in self.extract_agents(expected))
        gen_agents = set(a.lower() for a in self.extract_agents(generated))
        scores["agent_f1"] = self.compute_f1(exp_agents, gen_agents)

        # Temporal F1
        exp_temps = set(self.extract_temporal_operators(expected))
        gen_temps = set(self.extract_temporal_operators(generated))
        scores["temporal_f1"] = self.compute_f1(exp_temps, gen_temps)

        # Logical F1
        exp_logic = set(self.extract_logical_operators(expected))
        gen_logic = set(self.extract_logical_operators(generated))
        scores["logical_f1"] = self.compute_f1(exp_logic, gen_logic)

        # Syntax validity
        syntax = self.check_syntax(generated)
        scores["syntax_valid"] = float(syntax["all_valid"])

        # Overall score
        scores["overall_score"] = (
            0.15 * scores["exact_match"]
            + 0.15 * scores["agent_f1"]
            + 0.25 * scores["temporal_f1"]
            + 0.20 * scores["logical_f1"]
            + 0.25 * scores["syntax_valid"]
        )

        return scores

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
            scores = self.compute_scores(item["output"], generated)

            result = {
                "input": item["input"],
                "expected": item["output"],
                "generated": generated,
                "scores": scores,
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

        metrics = {
            "n_examples": n,
            "exact_match": np.mean([r["scores"]["exact_match"] for r in self.results]),
            "agent_f1": np.mean([r["scores"]["agent_f1"] for r in self.results]),
            "temporal_f1": np.mean([r["scores"]["temporal_f1"] for r in self.results]),
            "logical_f1": np.mean([r["scores"]["logical_f1"] for r in self.results]),
            "syntax_valid": np.mean(
                [r["scores"]["syntax_valid"] for r in self.results]
            ),
            "overall_score": np.mean(
                [r["scores"]["overall_score"] for r in self.results]
            ),
        }

        # Standard deviations
        metrics["exact_match_std"] = np.std(
            [r["scores"]["exact_match"] for r in self.results]
        )
        metrics["overall_score_std"] = np.std(
            [r["scores"]["overall_score"] for r in self.results]
        )

        return metrics
