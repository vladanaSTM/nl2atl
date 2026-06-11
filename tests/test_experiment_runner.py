import json

from src.constants import ModelType
from src.config import Config, ModelConfig, ExperimentCondition
from src.experiment.reporter import ExperimentReporter, sha256_file
from src.experiment.runner import ExperimentRunner


class _LengthTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_training_dataset_joins_all_outputs_into_one_target():
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
    # A multi-reading item is trained as a single target containing every
    # required formula, one per line, so the model learns to emit all readings
    # rather than choosing only one.
    assert len(dataset) == 1
    assert dataset[0]["prompt"].endswith("<|assistant|>\n")
    assert dataset[0]["completion"] == (
        "<<agent_1,agent_2>>F prevent_breach\n"
        "<<agent_1>>F prevent_breach_1 && <<agent_2>>F prevent_breach_2<|end|>"
    )


def test_training_dataset_rejects_truncated_completions():
    items = [
        {
            "id": "too-long",
            "input": " ".join(["agent"] * 1200),
            "outputs": [{"formula": "<<agent>>F safe"}],
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


def test_max_eval_samples_caps_test_set_with_stratified_coverage(tmp_path):
    from src.data_utils import default_stratum

    dataset_path = tmp_path / "dataset.json"
    data = []
    for i in range(20):
        data.append(
            {
                "id": f"single{i}",
                "input": f"single requirement {i}",
                "outputs": [{"formula": "<<A>>F p"}],
            }
        )
    for i in range(10):
        data.append(
            {
                "id": f"multi{i}",
                "input": f"multi requirement {i}",
                "outputs": [{"formula": "<<A>>F p"}, {"formula": "<<B>>G q"}],
            }
        )
    dataset_path.write_text(json.dumps(data), encoding="utf-8")

    config = Config(
        seed=42,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        augment_factor=1,
        max_eval_samples=2,
        data_path=str(dataset_path),
        output_dir=str(tmp_path / "out"),
    )

    runner = ExperimentRunner(config)

    # The smoke cap shrinks the evaluated test set and keeps both formula
    # structures so a tiny run still exercises single- and multi-answer output.
    assert len(runner.test_data) == 2
    assert {default_stratum(item) for item in runner.test_data} == {"single", "multi"}
    # Smoke results are routed into a dedicated subfolder so they never mix with
    # real prediction files.
    assert runner.reporter.predictions_subdir == "smoke_test"
    result_path = runner.reporter.get_result_path("gpt-5.4_baseline_zero_shot_seed42")
    assert result_path.parent == tmp_path / "out" / "model_predictions" / "smoke_test"


def test_reporter_writes_reproducible_split_manifest(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "train-1", "input": "a", "outputs": [{"formula": "<<A>>F p"}]},
                {"id": "test-1", "input": "b", "outputs": [{"formula": "<<B>>G q"}]},
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


def test_reporter_subdir_isolates_smoke_outputs(tmp_path):
    default_reporter = ExperimentReporter(output_dir=tmp_path)
    smoke_reporter = ExperimentReporter(
        output_dir=tmp_path, predictions_subdir="smoke_test"
    )

    assert default_reporter.get_result_path("run-1") == (
        tmp_path / "model_predictions" / "run-1.json"
    )
    assert smoke_reporter.get_result_path("run-1") == (
        tmp_path / "model_predictions" / "smoke_test" / "run-1.json"
    )

    config = Config(
        seed=42, train_size=0.7, val_size=0.1, test_size=0.2, augment_factor=1
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([{"id": "t1", "input": "a", "outputs": [{"formula": "<<A>>F p"}]}]),
        encoding="utf-8",
    )

    manifest_path = smoke_reporter.save_split_manifest(
        run_name="run-1",
        config=config,
        dataset_path=str(dataset_path),
        train_data=[],
        val_data=[],
        test_data=[{"id": "t1", "input": "a", "outputs": ["<<A>>F p"]}],
    )
    assert manifest_path.parent == tmp_path / "split_manifests" / "smoke_test"


def test_run_metadata_surfaces_content_filtered_examples(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([{"id": "t1", "input": "a", "outputs": [{"formula": "<<A>>F p"}]}]),
        encoding="utf-8",
    )
    config = Config(
        seed=42, train_size=0.7, val_size=0.1, test_size=0.2, augment_factor=1
    )
    model_config = ModelConfig(name="m", short_name="m")
    condition = ExperimentCondition(name="baseline_zero_shot", finetuned=False, few_shot=False)
    reporter = ExperimentReporter(output_dir=tmp_path)

    results = [
        {"id": "ok-1", "generated": "<<A>>F p", "exact_match": 1, "latency_ms": 1.0},
        {
            "id": "blocked-1",
            "generated": "",
            "exact_match": 0,
            "latency_ms": 1.0,
            "generation_error": "HTTP 500: content_filter",
        },
    ]

    metadata = reporter.build_run_metadata(
        config=config,
        run_name="run-1",
        model_config=model_config,
        condition=condition,
        effective_finetuned=False,
        dataset_path=str(dataset_path),
        total_samples=2,
        results=results,
    )

    assert metadata["successful_predictions"] == 1
    assert metadata["failed_predictions"] == 1
    assert metadata["content_filtered_predictions"] == 1
    assert metadata["content_filtered_ids"] == ["blocked-1"]
