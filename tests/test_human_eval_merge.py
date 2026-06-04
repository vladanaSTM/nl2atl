import csv
import json

from src.evaluation.human_eval_merge import MERGED_CSV_COLUMNS, merge_human_annotations


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

    annotation_a_path = tmp_path / "Francesco.csv"
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
                "annotator_id": "Francesco",
            }
        )

    annotation_b_path = tmp_path / "Marco.csv"
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
                "annotator_id": "Marco",
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

    annotation_path = tmp_path / "Francesco.csv"
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
                "annotator_id": "Francesco",
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

    annotation_a_path = tmp_path / "Francesco.csv"
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
                "annotator_id": "Francesco",
            }
        )

    annotation_b_path = tmp_path / "Marco.csv"
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
                "annotator_id": "Marco",
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
    assert item["francesco_correct"] == "yes"
    assert item["marco_correct"] == "no"
    assert item["francesco_matches_ds_v3_2"] == "yes"
    assert item["francesco_matches_gpt_5_2"] == "no"
    assert item["marco_matches_ds_v3_2"] == "no"
    assert item["marco_matches_gpt_5_2"] == "yes"
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
