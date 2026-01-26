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


def test_format_few_shot_prompt_includes_examples():
    prompt = few_shot._format_few_shot_section(
        [
            {"input": "A", "output": "<<A>>F p"},
            {"input": "B", "output": "<<B>>G q"},
        ]
    )
    assert "Example 1" in prompt
    assert "Input: A" in prompt
    assert "Output: <<A>>F p" in prompt


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


def test_format_prompt_qwen_tags():
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text=None,
        few_shot=False,
        model_type=ModelType.QWEN,
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>assistant" in prompt


def test_format_prompt_gemma_uses_tokenizer():
    tokenizer = _FakeTokenizer()
    prompt = few_shot.format_prompt(
        input_text="A",
        output_text=None,
        few_shot=False,
        model_type=ModelType.GEMMA,
        tokenizer=tokenizer,
    )
    assert prompt == "TEMPLATE"
    assert tokenizer.seen is not None
