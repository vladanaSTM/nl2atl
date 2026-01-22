from src import few_shot


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
