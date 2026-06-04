import json
import zipfile
from xml.etree import ElementTree as ET

from src.evaluation.human_eval_merge import load_human_annotations
from src.evaluation.human_eval_sample import (
    build_human_eval_sample,
    regenerate_annotator_workbooks_from_key,
)


def _xlsx_rows(path):
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in root.findall(".//x:sheetData/x:row", ns):
        rows.append(["".join(cell.itertext()) for cell in row.findall("x:c", ns)])
    return rows


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
    assert (output_dir / "annotations" / "annotator_1_blind.xlsx").exists()
    assert (output_dir / "annotations" / "annotator_2_blind.xlsx").exists()
    assert not (output_dir / "aaai_human_eval_sample_blind.csv").exists()
    assert not (output_dir / "aaai_disagreement_pool_blind.xlsx").exists()

    with zipfile.ZipFile(output_dir / "aaai_human_eval_sample_blind.xlsx") as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert '"yes,no"' in sheet_xml
    assert '"annotator_1,annotator_2"' in sheet_xml

    rows = _xlsx_rows(output_dir / "aaai_human_eval_sample_blind.xlsx")
    assert rows[0] == [
        "audit_id",
        "input",
        "gold_1",
        "gold_2",
        "prediction",
        "correct",
        "annotator_id",
    ]
    assert "gold_options" not in rows[0]
    assert rows[1][2]
    assert rows[1][3] == ""


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
    assert "gold_1" in blind_payload["annotations"][0]
    assert "gold_options" not in blind_payload["annotations"][0]
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
        annotator_choices=("annotator_1", "annotator_2"),
        backup_existing=True,
    )

    assert set(files) == {"annotator_1", "annotator_2"}
    assert (output_dir / "annotations" / "annotator_1_blind.xlsx").exists()
    assert list((output_dir / "annotations").glob("annotator_1_blind.*.bak.xlsx"))

    annotations = load_human_annotations(
        [output_dir / "annotations" / "annotator_2_blind.xlsx"]
    )
    first_annotation = next(iter(annotations.values()))[0]
    assert first_annotation["annotator_id"] == "annotator_2"
    assert first_annotation["correct"] == ""


def test_blind_workbook_splits_two_gold_options(tmp_path):
    eval_dir = tmp_path / "evaluated_datasets"
    for judge_name, decision in (("ds-v3.2", "yes"), ("gpt-5.2", "no")):
        path = eval_dir / judge_name / f"toy_run__judge-{judge_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_id": "toy_run",
                    "source_file": "toy_run.json",
                    "model_short": "toy-model",
                    "condition": "baseline_zero_shot",
                    "seed": 42,
                    "detailed_results": [
                        {
                            "id": "ex1",
                            "input": "Input with two gold options",
                            "gold": "<<A>>F p",
                            "gold_options": ["<<A>>F p", "<<B>>F q"],
                            "prediction": "<<A>>F p",
                            "correct": decision,
                            "decision_method": "llm",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    output_dir = tmp_path / "human_evaluation"
    build_human_eval_sample(
        eval_dir=eval_dir,
        output_dir=output_dir,
        quotas={"disagree_ds_yes_gpt_no": 1},
        sampling_seed=11,
    )

    rows = _xlsx_rows(output_dir / "annotations" / "annotator_1_blind.xlsx")
    assert rows[0][2:4] == ["gold_1", "gold_2"]
    assert rows[1][2:4] == ["<<A>>F p", "<<B>>F q"]
