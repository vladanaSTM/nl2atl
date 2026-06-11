import random

from src.models import few_shot
from src.constants import ModelType


class _FakeTokenizer:
    def __init__(self):
        self.seen = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "TEMPLATE"


def test_get_few_shot_examples_excludes_inputs():
    examples = few_shot.get_few_shot_examples(n=5, seed=123)
    assert len(examples) > 0
    first_input = examples[0]["input"]

    filtered = few_shot.get_few_shot_examples(
        n=5,
        seed=123,
        exclude_inputs=[first_input],
    )
    assert all(e["input"] != first_input for e in filtered)


def test_get_few_shot_examples_does_not_mutate_global_random():
    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    few_shot.get_few_shot_examples(n=3, seed=99)

    assert random.random() == expected


def test_get_few_shot_examples_defaults_to_all_curated():
    # The default (n=None) shows every curated exemplar in fixed curated order
    # so all distinct linguistic cases are demonstrated to the model.
    examples = few_shot.get_few_shot_examples()
    assert len(examples) == len(few_shot.FEW_SHOT_EXAMPLES)
    assert examples == few_shot.FEW_SHOT_EXAMPLES

    # n at least the pool size also returns the full curated pool in order.
    capped = few_shot.get_few_shot_examples(n=len(few_shot.FEW_SHOT_EXAMPLES) + 5)
    assert capped == few_shot.FEW_SHOT_EXAMPLES

    # The multi-reading (QSA) cases must always be present in the default pool.
    multi_reading = [
        e for e in examples if len(few_shot.get_all_output_formulas(e)) > 1
    ]
    assert len(multi_reading) >= 1


def test_format_few_shot_prompt_includes_examples():
    prompt = few_shot._format_few_shot_section(
        [
            {"input": "A", "outputs": [{"formula": "<<A>>F p"}]},
            {"input": "B", "outputs": [{"formula": "<<B>>G q"}]},
        ]
    )
    assert "Example 1" in prompt
    assert "Input: A" in prompt
    assert "Output:\n<<A>>F p" in prompt
    assert "Example 2" in prompt
    assert "Input: B" in prompt
    assert "Output:\n<<B>>G q" in prompt


def test_format_prompt_generic():
    prompt = few_shot.format_prompt(
        input_text="The agent can guarantee p",
        model_type="generic",
        few_shot=False,
    )
    assert "User:" in prompt
    assert "Assistant:" in prompt


def test_system_prompt_includes_few_shot_and_excludes():
    excluded = [
        "Agent A can guarantee that eventually p becomes true",
    ]
    prompt = few_shot.get_system_prompt(
        few_shot=True, num_examples=3, seed=1, exclude_inputs=excluded
    )
    assert "Here are some examples" in prompt
    assert excluded[0] not in prompt


def test_system_prompt_forbids_explanations_and_requires_all_readings():
    prompt = few_shot.get_system_prompt(few_shot=False)
    # Explanations and trailing prose are forbidden.
    assert "Do not include explanations" in prompt
    assert "Never append notes" in prompt
    # Single-reading inputs get one formula; QSA inputs get every reading.
    assert "return exactly one ATL/ATL* formula on a single line" in prompt
    assert "return all required ATL/ATL* formulas, one per line" in prompt


def test_format_prompt_qwen_tags():
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text=None,
        few_shot=False,
        model_type=ModelType.QWEN,
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>assistant" in prompt


def test_format_prompt_qwen_uses_tokenizer_template():
    tokenizer = _FakeTokenizer()
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text=None,
        few_shot=False,
        model_type=ModelType.QWEN,
        tokenizer=tokenizer,
    )
    assert prompt == "TEMPLATE"
    assert tokenizer.seen is not None
    assert tokenizer.seen["messages"][0]["role"] == "system"
    assert tokenizer.seen["messages"][1]["role"] == "user"
    assert tokenizer.seen["add_generation_prompt"] is True


def test_format_prompt_phi3_uses_tokenizer_template():
    tokenizer = _FakeTokenizer()
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text="<<A>>F p",
        few_shot=False,
        model_type=ModelType.PHI3,
        tokenizer=tokenizer,
    )
    assert prompt == "TEMPLATE"
    assert tokenizer.seen is not None
    assert tokenizer.seen["messages"][-1] == {
        "role": "assistant",
        "content": "<<A>>F p",
    }
    assert tokenizer.seen["add_generation_prompt"] is False


def test_format_prompt_phi3_fallback_uses_phi_tags():
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text="<<A>>F p",
        few_shot=False,
        model_type=ModelType.PHI3,
    )
    assert "<|system|>" in prompt
    assert "<|assistant|>" in prompt
    assert prompt.endswith("<<A>>F p<|end|>")


def test_format_prompt_mistral_uses_tokenizer_template():
    tokenizer = _FakeTokenizer()
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text=None,
        few_shot=False,
        model_type=ModelType.MISTRAL,
        tokenizer=tokenizer,
    )
    assert prompt == "TEMPLATE"
    assert tokenizer.seen is not None
    assert tokenizer.seen["messages"][0]["role"] == "user"
    assert tokenizer.seen["add_generation_prompt"] is True


def test_format_prompt_mistral_fallback_uses_inst_tags():
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text="<<A>>F p",
        few_shot=False,
        model_type=ModelType.MISTRAL,
    )
    assert "[INST]" in prompt
    assert "[/INST]" in prompt
    assert "<|im_start|>" not in prompt
    assert prompt.endswith("<<A>>F p</s>")
