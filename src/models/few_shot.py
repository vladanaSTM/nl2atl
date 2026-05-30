"""
Few-shot prompt management and formatting.
"""

import random
from typing import Any, List, Dict, Optional

from ..constants import ModelType
from ..data_utils import get_preferred_output

# Curated few-shot examples covering diverse ATL patterns
FEW_SHOT_EXAMPLES = [
    {
        "input": "They can guarantee that at the next step the alarm will be sent, the surveillance system and the operator.",
        "output": "<<System,Operator>>X alarm_sent",
    },
    {
        "input": "The gate, the machine can guarantee that it will open at the next step.",
        "output": "<<Machine>>X gate_open",
    },
    {
        "input": "Robot number 1 has a strategy to ensure that eventually position 3 holds, and robot number 2 does too.",
        "output": "<<Robot1>>F pos3 && <<Robot2>>F pos3",
    },
    {
        "input": "Every robot can guarantee that it will eventually reach a safe spot.",
        "output_1": "<<Robot1>>F at_safe_spot_1 && <<Robot2>>F at_safe_spot_2 && <<Robot3>>F at_safe_spot_3",
        "output_2": "<<Robot1,Robot2,Robot3>>F at_safe_spot",
    },
    {
        "input": "The diplomatic cable system can, but the encryption gateway cannot, guarantee that classified cables will never be routed publicly.",
        "output": "<<DiplomaticCableSystem>>G !classified_cables_routed_publicly && !<<EncryptionGateway>>G !classified_cables_routed_publicly",
    },
    {
        "input": "The user can guarantee that sooner or later the ticket will be printed.",
        "output": "<<User>>F ticket_printed",
    },
    {
        "input": "If we do not wish to fight, we can prevent the enemy from engaging us even though the lines of our encampment be merely traced out on the ground. All we need do is to throw something odd and unaccountable in his way.",
        "output": "<<We>>(!wish_to_fight -> F (throw_something_odd_in_his_way && G !enemy_engages_us))",
    },
]

SYSTEM_PROMPT_BASE = """You are an expert in Alternating-time Temporal Logic (ATL/ATL*).

Convert natural-language strategic specifications into well-formed ATL/ATL* formulas.

Output rules:
- Return exactly one ATL/ATL* formula on a single line.
- Do not include explanations, Markdown fences, labels, or natural-language text.

ATL/ATL* syntax rules:
- Agent coalition: <<Agent>> or <<Agent1,Agent2>>
- Coalitions may contain numbers or symbolic names, e.g. <<1>>, <<Machine>>, <<Robot,Operator>>
- Temporal operators: G (always), F (eventually), X (next), U (until), W (weak until), R (release)
- Logical operators: -> (implies), && (and), || (or), ! (not)
- Parentheses are required whenever an operator scopes over a compound formula.

Scope rules:
- The strategic operator <<A>> scopes over the whole temporal/path formula that follows it.
- Do not merge the strategic operator and the temporal operator conceptually.
- For example, write <<Machine>>G(paid -> ticket_printed), not <<Machine>>(G paid -> ticket_printed).
- If a temporal operator applies to a whole implication or conjunction, place the full compound formula inside its scope.
- Example: "always, if p then q" becomes G(p -> q), not G p -> q.
- Example: "next, p and q" becomes X(p && q), unless the sentence explicitly says that each is next separately.
- Do not introduce negation unless it is explicitly expressed in the natural-language input.
- Do not introduce temporal operators that are not licensed by the natural-language input.

Ambiguity rules:
- For VP ellipsis, repeat the full recovered strategic-temporal formula for the second agent.
- For Right Node Raising, attach the same shared right-peripheral objective to both coordinated strategic clauses.
- For quantifier-scope ambiguity, preserve the intended distributive or collective reading when specified.
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

    rng = random.Random(seed)
    return rng.sample(examples, min(n, len(examples)))


def _format_few_shot_section(examples: List[Dict]) -> str:
    """Format few-shot examples into a prompt section."""
    prompt = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        output = get_preferred_output(ex)
        if output is None:
            continue
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {output}\n\n"
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


def _chat_messages(
    system_prompt: str,
    input_text: str,
    output_text: Optional[str],
    *,
    merge_system_into_user: bool = False,
) -> List[Dict[str, str]]:
    """Build chat messages for tokenizer-owned templates."""
    user_content = f"Convert to ATL formula: {input_text}"
    if merge_system_into_user:
        messages = [
            {"role": "user", "content": f"{system_prompt.strip()}\n\n{user_content}"}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text})
    return messages


def _apply_chat_template(
    messages: List[Dict[str, str]],
    output_text: Optional[str],
    tokenizer: Optional[Any],
) -> Optional[str]:
    """Apply a tokenizer chat template when one is available."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=output_text is None,
    )


def _format_qwen(
    system_prompt: str,
    input_text: str,
    output_text: Optional[str],
    tokenizer: Optional[Any],
) -> str:
    """Format prompt for Qwen instruct and coder-instruct models."""
    messages = _chat_messages(system_prompt, input_text, output_text)
    templated = _apply_chat_template(messages, output_text, tokenizer)
    if templated is not None:
        return templated

    user_content = f"Convert to ATL formula: {input_text}"
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if output_text is not None:
        prompt += f"{output_text}<|im_end|>"
    return prompt


def _format_phi3(
    system_prompt: str,
    input_text: str,
    output_text: Optional[str],
    tokenizer: Optional[Any],
) -> str:
    """Format prompt for Phi-3/Phi-3.5 instruct models."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Convert to ATL formula: {input_text}"},
    ]
    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text})

    templated = _apply_chat_template(messages, output_text, tokenizer)
    if templated is not None:
        return templated

    prompt = (
        f"<|system|>\n{system_prompt}<|end|>\n"
        f"<|user|>\nConvert to ATL formula: {input_text}<|end|>\n"
        f"<|assistant|>\n"
    )
    if output_text is not None:
        prompt += f"{output_text}<|end|>"
    return prompt


def _format_mistral(
    system_prompt: str,
    input_text: str,
    output_text: Optional[str],
    tokenizer: Optional[Any],
) -> str:
    """Format prompt for Mistral instruct models."""
    messages = _chat_messages(
        system_prompt,
        input_text,
        output_text,
        merge_system_into_user=True,
    )
    templated = _apply_chat_template(messages, output_text, tokenizer)
    if templated is not None:
        return templated

    user_content = messages[0]["content"]
    prompt = f"<s>[INST] {user_content} [/INST]"
    if output_text is not None:
        prompt += f" {output_text}</s>"
    return prompt


def format_prompt(
    input_text: str,
    output_text: Optional[str] = None,
    few_shot: bool = False,
    num_examples: int = 5,
    model_type: str = ModelType.QWEN,
    exclude_inputs: Optional[List[str]] = None,
    tokenizer: Optional[Any] = None,
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
        tokenizer: Optional tokenizer used for native chat templates

    Returns:
        Formatted prompt string
    """
    system_prompt = get_system_prompt(
        few_shot=few_shot,
        num_examples=num_examples,
        exclude_inputs=exclude_inputs,
    )

    if model_type == ModelType.QWEN:
        return _format_qwen(system_prompt, input_text, output_text, tokenizer)

    if model_type == ModelType.PHI3:
        return _format_phi3(system_prompt, input_text, output_text, tokenizer)

    if model_type == ModelType.MISTRAL:
        return _format_mistral(system_prompt, input_text, output_text, tokenizer)

    user_content = f"Convert to ATL formula: {input_text}"
    prompt = f"System:\n{system_prompt}\n\nUser:\n{user_content}\n\nAssistant:\n"

    if output_text is not None:
        prompt += output_text

    return prompt
