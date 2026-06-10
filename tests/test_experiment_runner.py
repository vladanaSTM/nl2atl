import json

from src.constants import ModelType
from src.config import Config, ModelConfig
from src.experiment.reporter import ExperimentReporter, sha256_file
from src.experiment.runner import ExperimentRunner


class _LengthTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_training_dataset_uses_prompt_completion_pairs_for_each_output():
    items = [
        {
            "id": "dual",
            "input": "Every agent can eventually prevent a breach",
            "outputs": [
                "<<agent_1,agent_2>>F prevent_breach",
                "<<agent_1>>F prevent_breach_1 && <<agent_2>>F prevent_breach_2",
            ],
        }
    ]

    dataset = ExperimentRunner._training_dataset(
        items, model_type=ModelType.PHI3, tokenizer=None
    )

    assert dataset.column_names == ["prompt", "completion"]
    assert len(dataset) == 2
    assert all(
        dataset[index]["prompt"].endswith("<|assistant|>\n")
        for index in range(len(dataset))
    )
    assert dataset[0]["completion"] == "<<agent_1,agent_2>>F prevent_breach<|end|>"
    assert (
        dataset[1]["completion"]
        == "<<agent_1>>F prevent_breach_1 && <<agent_2>>F prevent_breach_2<|end|>"
    )


def test_training_dataset_rejects_truncated_completions():
    items = [
        {
            "id": "too-long",
            "input": " ".join(["agent"] * 1200),
            "output": "<<agent>>F safe",
        }
    ]

    try:
        ExperimentRunner._training_dataset(
            items,
            model_type=ModelType.PHI3,
            tokenizer=_LengthTokenizer(),
            max_seq_length=16,
        )
    except ValueError as exc:
        assert "max_seq_length" in str(exc)
        assert "too-long" in str(exc)
    else:
        raise AssertionError("Expected max_seq_length validation to fail")


def test_training_args_enable_completion_only_loss(tmp_path):
    runner = object.__new__(ExperimentRunner)
    runner.config = Config(seed=42, bf16=False, tf32=False, packing=False)

    args = runner._training_args(
        model_config=ModelConfig(name="model", short_name="model"),
        output_dir=tmp_path,
        max_steps=-1,
    )

    assert args.completion_only_loss is True


def test_reporter_writes_reproducible_split_manifest(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "train-1", "input": "a", "output": "<<A>>F p"},
                {"id": "test-1", "input": "b", "output": "<<B>>G q"},
            ]
        ),
        encoding="utf-8",
    )
    config = Config(
        seed=42,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        augment_factor=2,
    )
    reporter = ExperimentReporter(output_dir=tmp_path)

    manifest_path = reporter.save_split_manifest(
        run_name="run-1",
        config=config,
        dataset_path=str(dataset_path),
        train_data=[{"id": "train-1", "input": "a", "outputs": ["<<A>>F p"]}],
        val_data=[],
        test_data=[{"id": "test-1", "input": "b", "outputs": ["<<B>>G q"]}],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] == sha256_file(dataset_path)
    assert manifest["seed"] == 42
    assert manifest["counts"] == {
        "train": 1,
        "validation": 0,
        "test": 1,
    }
    assert manifest["train"][0]["id"] == "train-1"
    assert manifest["test"][0]["id"] == "test-1"
    assert "train_augmented" not in manifest
