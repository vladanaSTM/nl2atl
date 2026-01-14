"""
Few-shot prompt management.
"""

import random
from typing import List, Dict, Optional

# Few-shot examples (carefully curated, diverse patterns)
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


def get_few_shot_examples(
    n: int = 5, seed: int = None, exclude_inputs: List[str] = None
) -> List[Dict]:
    """
    Get n few-shot examples, optionally excluding certain inputs (to avoid data leakage).
    """
    examples = FEW_SHOT_EXAMPLES.copy()

    # Remove any examples that match test inputs
    if exclude_inputs:
        exclude_set = set(e.lower().strip() for e in exclude_inputs)
        examples = [
            e for e in examples if e["input"].lower().strip() not in exclude_set
        ]

    if seed is not None:
        random.seed(seed)

    return random.sample(examples, min(n, len(examples)))


def format_few_shot_prompt(examples: List[Dict]) -> str:
    """Format few-shot examples into a prompt section."""
    prompt = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n\n"
    return prompt


SYSTEM_PROMPT_BASE = """You are an expert in Alternating-Time Temporal Logic (ATL). 
Convert natural language specifications into ATL formulas. Output only the formula without any explanations.

ATL Syntax Rules:
- Agent coalition: <<Agent1,Agent2>> or <<Agent>>
- Temporal operators: G (always), F (eventually), X (next), U (until), W (weak until), R (release)
- Logical operators: -> (implies), & (and), | (or), ! (not)
- Parentheses for grouping: (condition -> result)
"""


def get_system_prompt(
    few_shot: bool = False,
    num_examples: int = 5,
    seed: int = 42,
    exclude_inputs: Optional[List[str]] = None,
) -> str:
    """Get system prompt, optionally with few-shot examples filtered from exclusions."""
    prompt = SYSTEM_PROMPT_BASE

    if few_shot:
        examples = get_few_shot_examples(
            n=num_examples,
            seed=seed,
            exclude_inputs=exclude_inputs,
        )
        prompt += "\n" + format_few_shot_prompt(examples)

    return prompt


def format_prompt(
    input_text: str,
    output_text: str = None,
    few_shot: bool = False,
    num_examples: int = 5,
    model_type: str = "qwen",
    exclude_inputs: Optional[List[str]] = None,
    tokenizer=None,
) -> str:
    """
    Format input for different model types.
    """
    system_prompt = get_system_prompt(
        few_shot=few_shot,
        num_examples=num_examples,
        exclude_inputs=exclude_inputs,
    )

    if model_type == "gemma":
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Convert to ATL formula: {input_text}",
            },
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

        prompt = (
            "<start_of_turn>system\n"
            f"{system_prompt}\n"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"Convert to ATL formula: {input_text}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        if output_text is not None:
            prompt += f"{output_text}<end_of_turn>"

        return prompt

    if model_type in ["qwen", "mistral"]:
        # ChatML format
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Convert to ATL formula: {input_text}<|im_end|>
<|im_start|>assistant
"""
    elif model_type == "phi3":
        # Phi-3 format
        prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
Convert to ATL formula: {input_text}<|end|>
<|assistant|>
"""
    elif model_type == "llama":
        # Llama format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Convert to ATL formula: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    else:
        # Generic format
        prompt = f"""{system_prompt}

User: Convert to ATL formula: {input_text}
Assistant: """

    if output_text:
        prompt += output_text
        if model_type in ["qwen", "mistral"]:
            prompt += "<|im_end|>"
        elif model_type == "phi3":
            prompt += "<|end|>"
        elif model_type == "llama":
            prompt += "<|eot_id|>"

    return prompt
