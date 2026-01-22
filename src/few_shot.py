"""
Few-shot prompt management and formatting.
"""

import random
from typing import List, Dict, Optional

from .constants import ModelType

# Curated few-shot examples covering diverse ATL patterns
FEW_SHOT_EXAMPLES = [
    {
        "input": "Agent A can guarantee that eventually p becomes true",
        "output": "<<A>>F p",
    },
    {
        "input": "Agents A and B can ensure that q is always true",
        "output": "<<A,B>>G q",
    },
    {
        "input": "The robot can guarantee that if error occurs then eventually it recovers",
        "output": "<<Robot>>G (error -> F recover)",
    },
    {
        "input": "The system keeps monitoring until the alarm is triggered",
        "output": "<<System>>(monitoring U alarm_triggered)",
    },
    {
        "input": "The controller can guarantee that at the next step the action happens",
        "output": "<<Controller>>X action",
    },
    {
        "input": "The drone will keep holding position until it receives land command",
        "output": "<<Drone>>(holding_position U land_command)",
    },
    {
        "input": "The security system guarantees that if intrusion detected then alarm sounds forever",
        "output": "<<Security>>G (intrusion_detected -> G alarm_on)",
    },
    {
        "input": "The coalition of robots can ensure the task is eventually completed",
        "output": "<<Robot1,Robot2>>F task_completed",
    },
]

SYSTEM_PROMPT_BASE = """You are an expert in Alternating-Time Temporal Logic (ATL). 
Convert natural language specifications into ATL formulas. Output only the formula without any explanations.

ATL Syntax Rules:
- Agent coalition: <<Agent1,Agent2>> or <<Agent>>
- Temporal operators: G (always), F (eventually), X (next), U (until), W (weak until), R (release)
- Logical operators: -> (implies), & (and), | (or), ! (not)
- Parentheses for grouping: (condition -> result)
"""


def get_few_shot_examples(
    n: int = 5,
    seed: Optional[int] = None,
    exclude_inputs: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Get n few-shot examples, optionally excluding certain inputs.

    Args:
        n: Number of examples to return
        seed: Random seed for reproducibility
        exclude_inputs: Inputs to exclude (for avoiding data leakage)

    Returns:
        List of example dictionaries
    """
    examples = FEW_SHOT_EXAMPLES.copy()

    if exclude_inputs:
        exclude_set = {e.lower().strip() for e in exclude_inputs}
        examples = [
            e for e in examples if e["input"].lower().strip() not in exclude_set
        ]

    if seed is not None:
        random.seed(seed)

    return random.sample(examples, min(n, len(examples)))


def _format_few_shot_section(examples: List[Dict]) -> str:
    """Format few-shot examples into a prompt section."""
    prompt = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n\n"
    return prompt


def get_system_prompt(
    few_shot: bool = False,
    num_examples: int = 5,
    seed: int = 42,
    exclude_inputs: Optional[List[str]] = None,
) -> str:
    """
    Get system prompt, optionally with few-shot examples.

    Args:
        few_shot: Whether to include few-shot examples
        num_examples: Number of examples to include
        seed: Random seed for example selection
        exclude_inputs: Inputs to exclude from examples

    Returns:
        Complete system prompt string
    """
    prompt = SYSTEM_PROMPT_BASE

    if few_shot:
        examples = get_few_shot_examples(
            n=num_examples,
            seed=seed,
            exclude_inputs=exclude_inputs,
        )
        prompt += "\n" + _format_few_shot_section(examples)

    return prompt


def _format_gemma(
    system_prompt: str,
    input_text: str,
    output_text: Optional[str],
    tokenizer: Optional[any],
) -> str:
    """Format prompt for Gemma models."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Convert to ATL formula: {input_text}"},
    ]

    add_generation_prompt = output_text is None
    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Manual fallback
    prompt = (
        f"<start_of_turn>system\n{system_prompt}\n<end_of_turn>\n"
        f"<start_of_turn>user\nConvert to ATL formula: {input_text}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    if output_text is not None:
        prompt += f"{output_text}<end_of_turn>"

    return prompt


def format_prompt(
    input_text: str,
    output_text: Optional[str] = None,
    few_shot: bool = False,
    num_examples: int = 5,
    model_type: str = ModelType.QWEN,
    exclude_inputs: Optional[List[str]] = None,
    tokenizer: Optional[any] = None,
) -> str:
    """
    Format input for different model types.

    Args:
        input_text: Natural language input
        output_text: Expected output (for training) or None (for inference)
        few_shot: Whether to include few-shot examples
        num_examples: Number of few-shot examples
        model_type: Model family type
        exclude_inputs: Inputs to exclude from few-shot examples
        tokenizer: Tokenizer for models that need it (e.g., Gemma)

    Returns:
        Formatted prompt string
    """
    system_prompt = get_system_prompt(
        few_shot=few_shot,
        num_examples=num_examples,
        exclude_inputs=exclude_inputs,
    )

    # Handle Gemma separately due to chat template needs
    if model_type == ModelType.GEMMA:
        return _format_gemma(system_prompt, input_text, output_text, tokenizer)

    # Build prompt based on model type
    user_content = f"Convert to ATL formula: {input_text}"

    if model_type in (ModelType.QWEN, ModelType.MISTRAL):
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        end_token = "<|im_end|>"

    elif model_type == ModelType.PHI3:
        prompt = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_content}<|end|>\n"
            f"<|assistant|>\n"
        )
        end_token = "<|end|>"

    elif model_type == ModelType.LLAMA:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        end_token = "<|eot_id|>"

    else:  # Generic
        prompt = f"{system_prompt}\n\nUser: {user_content}\nAssistant: "
        end_token = ""

    if output_text:
        prompt += output_text
        if end_token:
            prompt += end_token

    return prompt
