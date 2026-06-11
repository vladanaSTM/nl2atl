"""
Few-shot prompt management and formatting.
"""

import hashlib
import random
from typing import Any, List, Dict, Optional

from ..constants import ModelType

# Curated few-shot examples covering diverse ATL/ATL* patterns
# The schema mirrors the current dataset:
# each item has an "outputs" list; each output contains a "formula".
FEW_SHOT_EXAMPLES = [
    {
        "input": "They can guarantee that at the next step the alarm will be sent, the surveillance system and the operator.",
        "outputs": [
            {
                "formula": "<<System,Operator>>X alarm_sent"
            }
        ],
    },
    {
        "input": "The gate, the machine can guarantee that it will open at the next step.",
        "outputs": [
            {
                "formula": "<<Machine>>X gate_open"
            }
        ],
    },
    {
        "input": "Robot number 1 has a strategy to ensure that eventually position 3 holds, and robot number 2 does too.",
        "outputs": [
            {
                "formula": "<<Robot1>>F pos3 && <<Robot2>>F pos3"
            }
        ],
    },
    {
        "input": "Every robot can guarantee that it will eventually reach a safe spot.",
        "outputs": [
            {
                "formula": "<<Robot1>>F at_safe_spot_1 && <<Robot2>>F at_safe_spot_2 && <<Robot3>>F at_safe_spot_3"
            },
            {
                "formula": "<<Robot1,Robot2,Robot3>>F at_safe_spot"
            }
        ],
    },
    {
        "input": "The diplomatic cable system can, but the encryption gateway cannot, guarantee that classified cables will never be routed publicly.",
        "outputs": [
            {
                "formula": "<<DiplomaticCableSystem>>G !classified_cables_routed_publicly && !<<EncryptionGateway>>G !classified_cables_routed_publicly"
            }
        ],
    },
    {
        "input": "The user can guarantee that sooner or later the ticket will be printed.",
        "outputs": [
            {
                "formula": "<<User>>F ticket_printed"
            }
        ],
    },
    {
        "input": "If we do not wish to fight, we can prevent the enemy from engaging us even though the lines of our encampment be merely traced out on the ground. All we need do is to throw something odd and unaccountable in his way.",
        "outputs": [
            {
                "formula": "<<We>>(!wish_to_fight -> F (throw_something_odd_in_his_way && G !enemy_engages_us))"
            }
        ],
    },
]


SYSTEM_PROMPT_BASE = """You are an expert in Alternating-time Temporal Logic (ATL/ATL*).

Convert natural-language strategic specifications into well-formed ATL/ATL* formulas.

Output rules:
- Return only ATL/ATL* formula text.
- For ordinary single-reading inputs, return exactly one ATL/ATL* formula on a single line.
- For quantifier-scope ambiguity inputs with multiple admissible readings, return all required ATL/ATL* formulas, one per line.
- Do not merge multiple QSA readings into one conjunctive formula unless the natural-language input explicitly requires a conjunction.
- Do not include explanations, Markdown fences, labels, numbering, or natural-language text.
- Stop immediately after the formula or formulas. Never append notes beginning with words like "where", "or equivalently", "which means", or "represents".

ATL/ATL* syntax rules:
- Agent coalition: <<Agent>> or <<Agent1,Agent2>>
- Coalitions may contain numbers or symbolic names, e.g. <<1>>, <<Machine>>, <<Robot,Operator>>
- Temporal operators supported in this benchmark: G (always), F (eventually), X (next), U (until).
- Logical operators: -> (implies), && (and), || (or), ! (not)
- Parentheses are required whenever an operator scopes over a compound formula.
- ATL formulas are valid ATL* formulas. ATL* additionally allows more general nesting of temporal operators, such as XF p, XG !p, G(F p), or G(p -> F q), when licensed by the input.

Scope rules:
- The strategic operator <<A>> scopes over the whole formula that follows it.
- Do not merge the strategic operator and the temporal operator conceptually.
- For example, write <<Machine>>G(paid -> ticket_printed), not <<Machine>>(G paid -> ticket_printed).
- If a temporal operator applies to a whole implication or conjunction, place the full compound formula inside its scope.
- Example: "always, if p then q" becomes G(p -> q), not G p -> q.
- Example: "next, p and q" becomes X(p && q), unless the sentence explicitly says that each condition is next separately.
- Do not introduce negation unless it is explicitly expressed in the natural-language input.
- Do not introduce temporal operators that are not licensed by the natural-language input.

Ambiguity rules:
- For VP ellipsis, repeat the full recovered strategic-temporal formula for the second agent.
- For Right Node Raising, attach the same shared right-peripheral objective to both coordinated strategic clauses.
- For quantifier-scope ambiguity, produce all admissible readings as separate ATL/ATL* formulas.
- In QSA cases, do not choose only one reading. If both distributive and collective readings are admissible, include both.
- Keep distinct QSA readings on separate lines; do not collapse them into a single conjunction unless the input itself requires one conjunctive specification.
"""

def get_all_output_formulas(example: Dict[str, Any]) -> List[str]:
    """
    Return all expected ATL/ATL* formulas for an example.

    Dataset schema:
      "outputs": [{"formula": "..."}, ...]
    """
    outputs = example.get("outputs")
    if not isinstance(outputs, list):
        return []

    formulas = []
    for out in outputs:
        if isinstance(out, dict) and out.get("formula"):
            formulas.append(str(out["formula"]).strip())
        elif isinstance(out, str) and out.strip():
            formulas.append(out.strip())
    return formulas


def get_output_text(example: Dict[str, Any]) -> Optional[str]:
    """
    Format all required formulas as the assistant target.

    Single-output examples produce one line.
    Multi-output examples produce one formula per line.
    """
    formulas = get_all_output_formulas(example)
    if not formulas:
        return None
    return "\n".join(formulas)


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


def get_few_shot_example_id(example: Dict[str, Any]) -> str:
    """Return a stable identifier for a curated few-shot example."""
    if example.get("id"):
        return str(example["id"])

    output_text = get_output_text(example) or ""
    fingerprint = f"{example.get('input', '')}\n{output_text}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def _format_few_shot_section(examples: List[Dict]) -> str:
    """Format few-shot examples into a prompt section."""
    prompt = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        output = get_output_text(ex)
        if output is None:
            continue
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output:\n{output}\n\n"
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
