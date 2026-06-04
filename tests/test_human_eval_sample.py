import json
import zipfile

from src.evaluation.human_eval_merge import load_human_annotations
from src.evaluation.human_eval_sample import (
    build_human_eval_sample,
    regenerate_annotator_workbooks_from_key,
)


def _write_judge_file(path, judge_name, decisions):
    items = []
    for item_index, decision in enumerate(decisions, start=1):
        method = "exact" if item_index == 1 else "llm"
        items.append(
            {
                "id": f"ex{item_index}",
                "input": f"Input {item_index}",
                "gold": f"<<A>>F p{item_index}",
                "gold_options": [f"<<A>>F p{item_index}"],
                "prediction": f"<<A>>F p{item_index}",
                "correct": decision,
                "reasoning": f"{judge_name} reasoning {item_index}",
                "decision_method": method,
            }
        )

    payload = {
        "run_id": "toy_run",
        "source_file": "toy_run.json",
        "model_short": "toy-model",
        "condition": "baseline_zero_shot",
        "seed": 42,
        "finetuned": False,
        "few_shot": False,
        "detailed_results": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_human_eval_sample_writes_blind_and_keyed_files(tmp_path):
    eval_dir = tmp_path / "evaluated_datasets"
    _write_judge_file(
        eval_dir / "ds-v3.2" / "toy_run__judge-ds-v3.2.json",
        "ds-v3.2",
        ["yes", "yes", "no"],
    )
    _write_judge_file(
        eval_dir / "gpt-5.2" / "toy_run__judge-gpt-5.2.json",
        "gpt-5.2",
        ["yes", "no", "no"],
    )

    output_dir = tmp_path / "human_evaluation"
    metadata = build_human_eval_sample(
        eval_dir=eval_dir,
        output_dir=output_dir,
        quotas={
            "exact_match": 1,
            "disagree_ds_yes_gpt_no": 1,
            "llm_agree_no": 1,
        },
        sampling_seed=7,
    )

    key_payload = json.loads(
        (output_dir / "aaai_human_eval_sample_key.json").read_text(encoding="utf-8")
    )

    assert metadata["sample_size"] == 3
    assert len(key_payload["items"]) == 3
    assert "judge_decisions" in key_payload["items"][0]
    assert (output_dir / "aaai_human_eval_sample_blind.xlsx").exists()
    assert (output_dir / "annotations" / "Francesco_blind.xlsx").exists()
    assert (output_dir / "annotations" / "Marco_blind.xlsx").exists()
    assert not (output_dir / "aaai_human_eval_sample_blind.csv").exists()
    assert not (output_dir / "aaai_disagreement_pool_blind.xlsx").exists()

    with zipfile.ZipFile(output_dir / "aaai_human_eval_sample_blind.xlsx") as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert '"yes,no"' in sheet_xml
    assert '"Francesco,Marco"' in sheet_xml


def test_build_human_eval_sample_can_write_legacy_formats_and_pool(tmp_path):
    eval_dir = tmp_path / "evaluated_datasets"
    _write_judge_file(
        eval_dir / "ds-v3.2" / "toy_run__judge-ds-v3.2.json",
        "ds-v3.2",
        ["yes", "yes", "no"],
    )
    _write_judge_file(
        eval_dir / "gpt-5.2" / "toy_run__judge-gpt-5.2.json",
        "gpt-5.2",
        ["yes", "no", "no"],
    )

    output_dir = tmp_path / "human_evaluation"
    build_human_eval_sample(
        eval_dir=eval_dir,
        output_dir=output_dir,
        quotas={"disagree_ds_yes_gpt_no": 1, "llm_agree_no": 1},
        sampling_seed=8,
        write_disagreement_pool=True,
        write_legacy_formats=True,
    )

    blind_payload = json.loads(
        (output_dir / "aaai_human_eval_sample_blind.json").read_text(encoding="utf-8")
    )
    assert len(blind_payload["annotations"]) == 2
    assert "judge_decisions" not in blind_payload["annotations"][0]
    assert (output_dir / "aaai_disagreement_pool_blind.xlsx").exists()
    assert (output_dir / "aaai_disagreement_pool_blind.csv").exists()


def test_build_human_eval_sample_xlsx_can_be_read_by_merge(tmp_path):
    eval_dir = tmp_path / "evaluated_datasets"
    _write_judge_file(
        eval_dir / "ds-v3.2" / "toy_run__judge-ds-v3.2.json",
        "ds-v3.2",
        ["yes", "yes", "no"],
    )
    _write_judge_file(
        eval_dir / "gpt-5.2" / "toy_run__judge-gpt-5.2.json",
        "gpt-5.2",
        ["yes", "no", "no"],
    )

    output_dir = tmp_path / "human_evaluation"
    build_human_eval_sample(
        eval_dir=eval_dir,
        output_dir=output_dir,
        quotas={"disagree_ds_yes_gpt_no": 1, "llm_agree_no": 1},
        sampling_seed=9,
        write_disagreement_pool=False,
    )

    annotations = load_human_annotations(
        [output_dir / "aaai_human_eval_sample_blind.xlsx"]
    )
    assert set(annotations) == {"HEVAL-0001", "HEVAL-0002"}


def test_regenerate_annotator_workbooks_from_key(tmp_path):
    eval_dir = tmp_path / "evaluated_datasets"
    _write_judge_file(
        eval_dir / "ds-v3.2" / "toy_run__judge-ds-v3.2.json",
        "ds-v3.2",
        ["yes", "yes", "no"],
    )
    _write_judge_file(
        eval_dir / "gpt-5.2" / "toy_run__judge-gpt-5.2.json",
        "gpt-5.2",
        ["yes", "no", "no"],
    )

    output_dir = tmp_path / "human_evaluation"
    build_human_eval_sample(
        eval_dir=eval_dir,
        output_dir=output_dir,
        quotas={"disagree_ds_yes_gpt_no": 1, "llm_agree_no": 1},
        sampling_seed=10,
    )

    files = regenerate_annotator_workbooks_from_key(
        key_path=output_dir / "aaai_human_eval_sample_key.json",
        output_dir=output_dir,
        annotator_choices=("Francesco", "Marco"),
        backup_existing=True,
    )

    assert set(files) == {"Francesco", "Marco"}
    assert (output_dir / "annotations" / "Francesco_blind.xlsx").exists()
    assert list((output_dir / "annotations").glob("Francesco_blind.*.bak.xlsx"))

    annotations = load_human_annotations(
        [output_dir / "annotations" / "Marco_blind.xlsx"]
    )
    first_annotation = next(iter(annotations.values()))[0]
    assert first_annotation["annotator_id"] == "Marco"
    assert first_annotation["correct"] == ""
