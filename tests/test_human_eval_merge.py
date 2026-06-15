import csv
import json
from pathlib import Path

from src.evaluation.human_eval_merge import (
    MERGED_CSV_COLUMNS,
    build_merged_csv_columns,
    merge_human_annotations,
)
from src.evaluation.judge_agreement import generate_agreement_report_with_human


def test_merge_human_annotations_deanonymizes_by_audit_id(tmp_path):
    key_path = tmp_path / "key.json"
    key_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "audit_id": "HEVAL-0001",
                        "model_short": "mistral",
                        "condition": "finetuned_zero_shot",
                        "seed": 42,
                        "primary_stratum": "llm_agree_yes",
                        "judge_decisions": {
                            "ds-v3.2": {"correct": "yes"},
                            "gpt-5.2": {"correct": "yes"},
                        },
                        "input": "Input",
                        "gold": "<<A>>F p",
                        "gold_options": ["<<A>>F p"],
                        "prediction": "<<A>>F p",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotation_a_path = tmp_path / "annotator_1.csv"
    with open(annotation_a_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["audit_id", "correct", "annotator_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "HEVAL-0001",
                "correct": "yes",
                "annotator_id": "annotator_1",
            }
        )

    annotation_b_path = tmp_path / "annotator_2.csv"
    with open(annotation_b_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["audit_id", "correct", "annotator_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "HEVAL-0001",
                "correct": "yes",
                "annotator_id": "annotator_2",
            }
        )

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_a_path, annotation_b_path],
        output_dir=tmp_path / "merged",
    )

    merged = json.loads(
        (tmp_path / "merged" / "aaai_human_eval_merged.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["summary"]["n_items_with_annotations"] == 1
    assert result["summary"]["n_items_with_final_label"] == 1
    assert result["summary"]["n_items_needing_adjudication"] == 0
    assert merged["items"][0]["model_short"] == "mistral"
    assert merged["items"][0]["human_consensus_correct"] == "yes"
    assert merged["items"][0]["human_final_correct"] == "yes"
    assert merged["items"][0]["human_matches_ds_v3_2"] == "yes"
    assert merged["items"][0]["human_matches_gpt_5_2"] == "yes"
    assert merged["items"][0]["n_human_matches_ds_v3_2"] == 2
    assert merged["items"][0]["human_match_rate_ds_v3_2"] == 1.0
    assert merged["items"][0]["gold_1"] == "<<A>>F p"
    assert merged["items"][0]["gold_2"] == ""
    assert "gold_options" not in merged["items"][0]
    assert "human_reasoning" not in merged["items"][0]
    assert "human_annotations" not in merged["items"][0]

    with open(
        tmp_path / "merged" / "aaai_human_eval_merged.csv",
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        row = next(csv.DictReader(csv_file))
    assert list(row) == MERGED_CSV_COLUMNS
    assert "human_reasoning" not in row
    assert "gold_options" not in row
    assert row["human_matches_ds_v3_2"] == "yes"


def test_merge_ignores_blank_template_rows(tmp_path):
    key_path = tmp_path / "key.json"
    key_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "audit_id": "HEVAL-0001",
                        "model_short": "mistral",
                        "condition": "finetuned_zero_shot",
                        "seed": 42,
                        "primary_stratum": "llm_agree_yes",
                        "judge_decisions": {
                            "ds-v3.2": {"correct": "yes"},
                            "gpt-5.2": {"correct": "yes"},
                        },
                        "input": "Input",
                        "gold": "<<A>>F p",
                        "gold_options": ["<<A>>F p"],
                        "prediction": "<<A>>F p",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotation_path = tmp_path / "annotator_1.csv"
    with open(annotation_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["audit_id", "correct", "annotator_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "HEVAL-0001",
                "correct": "",
                "annotator_id": "annotator_1",
            }
        )

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_path],
        output_dir=tmp_path / "merged",
    )

    merged = json.loads(
        (tmp_path / "merged" / "aaai_human_eval_merged.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["summary"]["n_items_with_annotations"] == 0
    assert merged["items"][0]["human_status"] == "unannotated"
    assert merged["items"][0]["n_human_labels"] == 0
    assert merged["items"][0]["annotator_ids"] == []
    assert merged["items"][0]["human_final_correct"] == "unannotated"
    assert merged["items"][0]["human_matches_ds_v3_2"] == "unannotated"


def test_merge_marks_human_disagreements_for_adjudication(tmp_path):
    key_path = tmp_path / "key.json"
    key_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "audit_id": "HEVAL-0002",
                        "model_short": "gpt-5.4",
                        "condition": "baseline_zero_shot",
                        "seed": 43,
                        "primary_stratum": "disagree_ds_yes_gpt_no",
                        "sampling_weight_primary": 6.823353,
                        "judge_decisions": {
                            "ds-v3.2": {"correct": "yes"},
                            "gpt-5.2": {"correct": "no"},
                        },
                        "input": "Input",
                        "gold": "<<A>>F p",
                        "gold_options": ["<<A>>F p"],
                        "prediction": "<<B>>F q",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotation_a_path = tmp_path / "annotator_1.csv"
    with open(annotation_a_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["audit_id", "correct", "annotator_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "HEVAL-0002",
                "correct": "yes",
                "annotator_id": "annotator_1",
            }
        )

    annotation_b_path = tmp_path / "annotator_2.csv"
    with open(annotation_b_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["audit_id", "correct", "annotator_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "HEVAL-0002",
                "correct": "no",
                "annotator_id": "annotator_2",
            }
        )

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_a_path, annotation_b_path],
        output_dir=tmp_path / "merged",
    )

    merged = json.loads(
        (tmp_path / "merged" / "aaai_human_eval_merged.json").read_text(
            encoding="utf-8"
        )
    )
    item = merged["items"][0]
    assert result["summary"]["n_items_needing_adjudication"] == 1
    assert result["summary"]["n_items_with_final_label"] == 0
    assert result["summary"]["llm_individual_human_agreement"]["ds-v3.2"] == {
        "n": 2,
        "matches": 1,
        "agreement_rate": 0.5,
    }
    assert item["human_status"] == "disagreement"
    assert item["needs_adjudication"] == "yes"
    assert item["human_consensus_correct"] == "no_consensus"
    assert item["human_final_correct"] == "pending_adjudication"
    assert item["human_matches_ds_v3_2"] == "pending_adjudication"
    assert item["human_matches_gpt_5_2"] == "pending_adjudication"
    assert item["annotator_1_correct"] == "yes"
    assert item["annotator_2_correct"] == "no"
    assert item["annotator_1_matches_ds_v3_2"] == "yes"
    assert item["annotator_1_matches_gpt_5_2"] == "no"
    assert item["annotator_2_matches_ds_v3_2"] == "no"
    assert item["annotator_2_matches_gpt_5_2"] == "yes"
    assert item["n_human_matches_ds_v3_2"] == 1
    assert item["n_human_matches_gpt_5_2"] == 1
    assert item["human_match_rate_ds_v3_2"] == 0.5
    assert item["human_match_rate_gpt_5_2"] == 0.5

    with open(
        tmp_path / "merged" / "aaai_human_eval_merged.csv",
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["human_final_correct"] == "pending_adjudication"
    assert row["human_match_rate_ds_v3_2"] == "0.5"


def _key_item(
    audit_id, prediction, ds, gpt, gold="<<A>>F p", source_file="run.json", input_text=None
):
    return {
        "audit_id": audit_id,
        "model_short": "mistral",
        "condition": "finetuned_zero_shot",
        "seed": 42,
        "primary_stratum": "disagree_ds_yes_gpt_no",
        "judge_decisions": {
            "ds-v3.2": {"correct": ds},
            "gpt-5.2": {"correct": gpt},
        },
        "input": input_text if input_text is not None else f"Input {audit_id}",
        "gold": gold,
        "gold_options": [gold],
        "prediction": prediction,
        "source_file": source_file,
    }


def _write_key(path, items):
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


def _write_annotator_csv(path, rows, annotator_id):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["audit_id", "correct", "annotator_id"]
        )
        writer.writeheader()
        for audit_id, correct in rows:
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "correct": correct,
                    "annotator_id": annotator_id,
                }
            )


def test_merge_reports_chance_corrected_kappa(tmp_path):
    key_path = tmp_path / "key.json"
    _write_key(
        key_path,
        [
            _key_item("HEVAL-0001", "<<A>>F p1", "yes", "yes"),
            _key_item("HEVAL-0002", "<<A>>F p2", "no", "no"),
            _key_item("HEVAL-0003", "<<A>>F p3", "yes", "no"),
            _key_item("HEVAL-0004", "<<A>>F p4", "no", "yes"),
        ],
    )

    # Annotators agree perfectly with each other on all four items.
    annotator_labels = [
        ("HEVAL-0001", "yes"),
        ("HEVAL-0002", "no"),
        ("HEVAL-0003", "yes"),
        ("HEVAL-0004", "no"),
    ]
    annotation_a = tmp_path / "annotator_1.csv"
    annotation_b = tmp_path / "annotator_2.csv"
    _write_annotator_csv(annotation_a, annotator_labels, "annotator_1")
    _write_annotator_csv(annotation_b, annotator_labels, "annotator_2")

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_a, annotation_b],
        output_dir=tmp_path / "merged",
    )

    reliability = result["summary"]["human_human_reliability"]
    assert reliability["n_annotators"] == 2
    assert reliability["cohen_kappa"] == 1.0
    assert reliability["fleiss_kappa"] == 1.0
    assert reliability["krippendorff_alpha"] == 1.0
    assert reliability["cohen_kappa_interpretation"] == "Almost Perfect"

    llm_reliability = result["summary"]["llm_human_reliability"]
    # ds-v3.2 matches the human gold on every item.
    assert llm_reliability["ds-v3.2"]["accuracy"] == 1.0
    assert llm_reliability["ds-v3.2"]["cohen_kappa"] == 1.0
    # gpt-5.2 agrees on two of four; chance-corrected kappa collapses to 0.
    assert llm_reliability["gpt-5.2"]["accuracy"] == 0.5
    assert llm_reliability["gpt-5.2"]["cohen_kappa"] == 0.0
    assert llm_reliability["gpt-5.2"]["n"] == 4


def test_build_merged_csv_columns_scales_to_n_annotators():
    columns = build_merged_csv_columns(
        ["annotator_1", "annotator_2", "annotator_3"], ["ds-v3.2", "gpt-5.2"]
    )
    assert "annotator_3_correct" in columns
    assert "annotator_3_matches_ds_v3_2" in columns
    assert "annotator_3_matches_gpt_5_2" in columns
    # Two-annotator default is unchanged for backward compatibility.
    assert build_merged_csv_columns(
        ["annotator_1", "annotator_2"], ["ds-v3.2", "gpt-5.2"]
    ) == MERGED_CSV_COLUMNS


def test_merge_supports_three_annotators(tmp_path):
    key_path = tmp_path / "key.json"
    _write_key(
        key_path,
        [
            _key_item("HEVAL-0001", "<<A>>F p1", "yes", "yes"),
            _key_item("HEVAL-0002", "<<A>>F p2", "no", "no"),
        ],
    )

    labels = [("HEVAL-0001", "yes"), ("HEVAL-0002", "no")]
    paths = []
    for annotator_id in ("annotator_1", "annotator_2", "annotator_3"):
        path = tmp_path / f"{annotator_id}.csv"
        _write_annotator_csv(path, labels, annotator_id)
        paths.append(path)

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=paths,
        output_dir=tmp_path / "merged",
    )

    assert result["summary"]["annotators"] == [
        "annotator_1",
        "annotator_2",
        "annotator_3",
    ]
    # Cohen's kappa is undefined for three raters; Fleiss generalizes.
    assert result["summary"]["human_human_reliability"]["cohen_kappa"] is None
    assert result["summary"]["human_human_reliability"]["fleiss_kappa"] == 1.0

    merged = json.loads(
        (tmp_path / "merged" / "aaai_human_eval_merged.json").read_text(
            encoding="utf-8"
        )
    )
    item = merged["items"][0]
    assert item["annotator_3_correct"] == "yes"
    assert "annotator_3_matches_ds_v3_2" in item

    with open(
        tmp_path / "merged" / "aaai_human_eval_merged.csv",
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        header = next(csv.reader(csv_file))
    assert "annotator_3_correct" in header
    assert "annotator_3_matches_gpt_5_2" in header


def test_merge_writes_adjudicated_human_gold_adapter(tmp_path):
    key_path = tmp_path / "key.json"
    _write_key(
        key_path,
        [
            _key_item("HEVAL-0001", "<<A>>F p1", "yes", "yes"),
            _key_item("HEVAL-0002", "<<A>>F p2", "no", "yes"),
        ],
    )

    labels = [("HEVAL-0001", "yes"), ("HEVAL-0002", "no")]
    annotation_a = tmp_path / "annotator_1.csv"
    annotation_b = tmp_path / "annotator_2.csv"
    _write_annotator_csv(annotation_a, labels, "annotator_1")
    _write_annotator_csv(annotation_b, labels, "annotator_2")

    result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_a, annotation_b],
        output_dir=tmp_path / "merged",
    )

    adapter_path = result["files"]["adjudicated_human_gold"]
    payload = json.loads(open(adapter_path, encoding="utf-8").read())
    assert payload["human_label"] == "human"
    assert payload["n_items"] == 2
    first = payload["items"][0]
    assert first["source_file"] == "run.json"
    assert first["gold"] == "<<A>>F p"
    assert first["correct"] in {"yes", "no"}
    assert "input" in first and "prediction" in first


def _write_llm_judge_file(path, source_file, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": source_file.split(".")[0],
                "source_file": source_file,
                "model_short": "mistral",
                "condition": "finetuned_zero_shot",
                "seed": 42,
                "detailed_results": [
                    {
                        "id": item["id"],
                        "input": item["input"],
                        "gold": item["gold"],
                        "gold_options": [item["gold"]],
                        "prediction": item["prediction"],
                        "correct": item["correct"],
                        "decision_method": "llm",
                    }
                    for item in items
                ],
            }
        ),
        encoding="utf-8",
    )


def test_adjudicated_adapter_feeds_human_comparison(tmp_path):
    # Four aligned items judged by both LLM judges.
    items = [
        {"id": "ex1", "input": "I1", "gold": "<<A>>F p1", "prediction": "<<A>>F p1"},
        {"id": "ex2", "input": "I2", "gold": "<<A>>F p2", "prediction": "<<A>>F p2"},
        {"id": "ex3", "input": "I3", "gold": "<<A>>F p3", "prediction": "<<B>>F p3"},
        {"id": "ex4", "input": "I4", "gold": "<<A>>F p4", "prediction": "<<B>>F p4"},
    ]
    ds_decisions = ["yes", "no", "yes", "no"]
    gpt_decisions = ["yes", "no", "no", "yes"]

    eval_dir = tmp_path / "evaluated_datasets"
    _write_llm_judge_file(
        eval_dir / "ds-v3.2" / "run__judge-ds-v3.2.json",
        "run.json",
        [{**item, "correct": decision} for item, decision in zip(items, ds_decisions)],
    )
    _write_llm_judge_file(
        eval_dir / "gpt-5.2" / "run__judge-gpt-5.2.json",
        "run.json",
        [{**item, "correct": decision} for item, decision in zip(items, gpt_decisions)],
    )

    key_path = tmp_path / "key.json"
    _write_key(
        key_path,
        [
            _key_item(
                f"HEVAL-000{index + 1}",
                item["prediction"],
                ds_decisions[index],
                gpt_decisions[index],
                gold=item["gold"],
                source_file="run.json",
                input_text=item["input"],
            )
            for index, item in enumerate(items)
        ],
    )

    human_labels = [
        ("HEVAL-0001", "yes"),
        ("HEVAL-0002", "no"),
        ("HEVAL-0003", "yes"),
        ("HEVAL-0004", "no"),
    ]
    annotation_a = tmp_path / "annotator_1.csv"
    annotation_b = tmp_path / "annotator_2.csv"
    _write_annotator_csv(annotation_a, human_labels, "annotator_1")
    _write_annotator_csv(annotation_b, human_labels, "annotator_2")

    merge_result = merge_human_annotations(
        key_path=key_path,
        annotation_paths=[annotation_a, annotation_b],
        output_dir=tmp_path / "merged",
    )
    adapter_path = merge_result["files"]["adjudicated_human_gold"]

    report = generate_agreement_report_with_human(
        eval_dir=eval_dir,
        human_annotations_path=Path(adapter_path),
        output_path=tmp_path / "agreement_report.json",
    )

    human_comparison = report["human_comparison"]
    assert human_comparison is not None
    per_judge = human_comparison["per_judge"]
    # The human gold aligns with every judged item.
    assert per_judge["ds-v3.2"]["n_common"] == 4
    assert per_judge["ds-v3.2"]["accuracy"] == 1.0
    assert per_judge["ds-v3.2"]["cohen_kappa"] == 1.0
    assert per_judge["gpt-5.2"]["accuracy"] == 0.5
    assert "cohen_kappa_interpretation" in per_judge["ds-v3.2"]

