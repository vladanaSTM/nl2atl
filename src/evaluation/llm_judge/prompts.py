"""Prompt templates and formatting for LLM judge."""

from dataclasses import dataclass
from typing import Any, Optional

PROMPT_VERSION = "v1.1"

JUDGE_PROMPT_TEMPLATE = (
    "You are an expert judge for ATL (Alternating-time Temporal Logic) formulas.\n"
    "Decide whether the prediction is semantically correct ATL for the given natural-language input.\n"
    "Some examples have multiple acceptable gold formulas. Mark the prediction correct if it is semantically equivalent to any one gold formula.\n"
    "Do not require the prediction to satisfy every listed gold formula when alternatives are provided.\n"
    "Be strict about meaning: incorrect if coalition/agent set, temporal operator (X/F/G/U),\n"
    "polarity (!p vs p), or connective (|| vs &&) changes the expressed property.\n\n"
    "Return ONLY machine-parseable JSON with keys correct and reasoning:\n"
    '{{ "correct": "yes" | "no", "reasoning": "..." }}\n\n'
    "Few-shot examples:\n"
    "Example 1 (correct despite deviation)\n"
    "input: The collaborative robot can guarantee that it will keep running the cycle until a stop is requested.\n"
    "gold: <<Cobot>>(cycle_running U stop_requested)\n"
    "prediction: <<CollaborativeRobot>>(running_cycle U stop_requested)\n"
    'output: {{ "correct": "yes", "reasoning": "Same coalition intent and same until structure; predicates are clear aliases from the sentence." }}\n\n'
    "Example 2 (correct despite deviation)\n"
    "input: The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.\n"
    "gold: <<Machine>>G (paid -> X ticket_printed)\n"
    "prediction: <<Machine>>G (!paid || X ticket_printed)\n"
    'output: {{ "correct": "yes", "reasoning": "Implication rewrite preserves meaning; same coalition and temporal structure." }}\n\n'
    "Example 3 (correct despite deviation)\n"
    "input: The user can guarantee that at the next step either a card or cash will be inserted.\n"
    "gold: <<User>>X (card_inserted || cash_inserted)\n"
    "prediction: <<User>>X (cash_inserted || card_inserted)\n"
    'output: {{ "correct": "yes", "reasoning": "Disjunction order doesn\'t matter; same agent and X." }}\n\n'
    "Example 4 (incorrect: wrong temporal operator)\n"
    "input: The user can guarantee that at the next step either a card or cash will be inserted.\n"
    "gold: <<User>>X (card_inserted || cash_inserted)\n"
    "prediction: <<User>>F (card_inserted || cash_inserted)\n"
    'output: {{ "correct": "no", "reasoning": "F allows it eventually, not necessarily next step X." }}\n\n'
    "Example 5 (incorrect: wrong agent)\n"
    "input: The machine can guarantee that if the payment has been completed, then at the next step it will print the ticket.\n"
    "gold: <<Machine>>G (paid -> X ticket_printed)\n"
    "prediction: <<User>>G (paid -> X ticket_printed)\n"
    'output: {{ "correct": "no", "reasoning": "Coalition changed; ability attributed to wrong actor." }}\n\n'
    "Example 6 (incorrect: polarity flipped)\n"
    "input: The controller can guarantee that the door is never open.\n"
    "gold: <<Controller>>G !door_open\n"
    "prediction: <<Controller>>G door_open\n"
    'output: {{ "correct": "no", "reasoning": "Negation flipped; expresses the opposite." }}\n\n'
    "Now evaluate:\n"
    "input: {input_text}\n"
    "gold: {gold}\n"
    "prediction: {prediction}\n"
    "output:"
)


@dataclass
class JudgePromptConfig:
    """Prompt configuration for judge evaluation."""

    template: str = JUDGE_PROMPT_TEMPLATE


def format_gold_options(gold: Any) -> str:
    """Format one or more acceptable gold formulas for the judge prompt."""
    if gold is None:
        options = []
    elif isinstance(gold, (list, tuple)):
        options = [str(item).strip() for item in gold if str(item).strip()]
    else:
        options = [str(gold).strip()] if str(gold).strip() else []

    if len(options) <= 1:
        return options[0] if options else ""

    return "\n".join(f"{index}. {option}" for index, option in enumerate(options, 1))


def format_judge_prompt(
    input_text: str,
    gold: Any,
    prediction: str,
    config: Optional[JudgePromptConfig] = None,
) -> str:
    """Format a judge evaluation prompt."""
    cfg = config or JudgePromptConfig()
    return cfg.template.format(
        input_text=input_text,
        gold=format_gold_options(gold),
        prediction=prediction,
    )
