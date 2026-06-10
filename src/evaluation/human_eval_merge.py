"""Merge blind human annotations with the private human-eval key."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from xml.etree import ElementTree as ET

from ..infra.io import load_json, save_json

KNOWN_ANNOTATORS = ("annotator_1", "annotator_2")

MERGED_CSV_COLUMNS = [
    "audit_id",
    "human_status",
    "needs_adjudication",
    "n_human_labels",
    "annotator_1_correct",
    "annotator_2_correct",
    "human_consensus_correct",
    "human_final_correct",
    "human_labels",
    "annotator_ids",
    "n_human_yes",
    "n_human_no",
    "ds_v3_2_correct",
    "gpt_5_2_correct",
    "llm_judges_agree",
    "human_matches_ds_v3_2",
    "human_matches_gpt_5_2",
    "n_human_matches_ds_v3_2",
    "n_human_matches_gpt_5_2",
    "human_match_rate_ds_v3_2",
    "human_match_rate_gpt_5_2",
    "annotator_1_matches_ds_v3_2",
    "annotator_1_matches_gpt_5_2",
    "annotator_2_matches_ds_v3_2",
    "annotator_2_matches_gpt_5_2",
    "model_short",
    "condition",
    "seed",
    "primary_stratum",
    "sampling_weight_primary",
    "input",
    "gold_1",
    "gold_2",
    "prediction",
    "source_file",
    "item_id",
]


def _normalize_correct(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    value_str = str(value or "").strip().lower()
    if value_str in {"yes", "y", "true", "1", "correct"}:
        return "yes"
    if value_str in {"no", "n", "false", "0", "incorrect"}:
        return "no"
    return ""


def _annotation_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as csv_file:
            return [dict(row) for row in csv.DictReader(csv_file)]
    if suffix == ".xlsx":
        return _xlsx_rows(path)

    data = load_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = data.get("annotations") or data.get("items") or data.get("data")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported annotation format: {path}")


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0
    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - 64
    return index - 1


def _xlsx_rows(path: Path) -> List[Dict[str, Any]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml")
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("x:si", ns):
                shared_strings.append("".join(si.itertext()))

    root = ET.fromstring(sheet_xml)
    rows: List[List[str]] = []
    for row in root.findall(".//x:sheetData/x:row", ns):
        values: List[str] = []
        for cell in row.findall("x:c", ns):
            column = _column_index(cell.attrib.get("r", ""))
            while len(values) <= column:
                values.append("")
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(cell.findtext("x:is/x:t", default="", namespaces=ns))
            else:
                raw_value = cell.findtext("x:v", default="", namespaces=ns)
                if cell_type == "s" and raw_value:
                    value = shared_strings[int(raw_value)]
                else:
                    value = raw_value
            values[column] = value
        rows.append(values)

    if not rows:
        return []
    header = rows[0]
    return [
        {
            header[index]: row[index] if index < len(row) else ""
            for index in range(len(header))
        }
        for row in rows[1:]
    ]


def load_human_annotations(
    annotation_paths: Sequence[Path],
) -> Dict[str, List[Dict[str, Any]]]:
    annotations_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for path in annotation_paths:
        for row in _annotation_rows(path):
            audit_id = str(row.get("audit_id") or "").strip()
            if not audit_id:
                continue
            annotator_id = str(row.get("annotator_id") or "").strip() or path.stem
            normalized_row = {
                "annotator_id": annotator_id,
                "correct": _normalize_correct(row.get("correct")),
            }
            annotations_by_id.setdefault(audit_id, []).append(normalized_row)
    return annotations_by_id


def _human_summary(annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final_annotator_ids = {"adjudicated", "adjudicator", "consensus", "final"}
    final_annotations = [
        annotation
        for annotation in annotations
        if str(annotation.get("annotator_id") or "").strip().lower()
        in final_annotator_ids
    ]
    independent_annotations = [
        annotation for annotation in annotations if annotation not in final_annotations
    ]

    labeled_annotations = [
        annotation
        for annotation in independent_annotations
        if annotation.get("correct", "") in {"yes", "no"}
    ]
    labels = [annotation.get("correct", "") for annotation in labeled_annotations]
    label_counts = Counter(labels)
    labels_by_annotator = {
        str(annotation.get("annotator_id") or ""): annotation.get("correct", "")
        for annotation in labeled_annotations
    }
    final_labels = [
        annotation.get("correct", "")
        for annotation in final_annotations
        if annotation.get("correct", "") in {"yes", "no"}
    ]

    if not labels:
        status = "unannotated"
        consensus = "unannotated"
    elif len(labels) == 1:
        status = "single_annotation"
        consensus = "single_annotation"
    elif len(label_counts) == 1:
        status = "agreement"
        consensus = labels[0]
    else:
        status = "disagreement"
        consensus = "no_consensus"

    if final_labels:
        human_final_correct = final_labels[-1]
    elif status == "agreement":
        human_final_correct = consensus
    elif status == "disagreement":
        human_final_correct = "pending_adjudication"
    elif status == "single_annotation":
        human_final_correct = "pending_second_annotation"
    else:
        human_final_correct = "unannotated"

    return {
        "human_status": status,
        "needs_adjudication": status == "disagreement" and not final_labels,
        "human_consensus_correct": consensus,
        "human_final_correct": human_final_correct,
        "human_label_counts": dict(label_counts),
        "human_labels": labels,
        "labels_by_annotator": labels_by_annotator,
        "annotator_ids": [
            annotation.get("annotator_id") for annotation in labeled_annotations
        ],
        "n_human_labels": len(labels),
    }


def _judge_correct(item: Mapping[str, Any], judge_name: str) -> str:
    decisions = item.get("judge_decisions") or {}
    judge = decisions.get(judge_name) or {}
    return _normalize_correct(judge.get("correct"))


def _label_match(human_label: str, judge_label: str) -> str:
    if human_label in {
        "unannotated",
        "single_annotation",
        "pending_second_annotation",
        "no_consensus",
        "pending_adjudication",
    }:
        return human_label
    if human_label not in {"yes", "no"} or judge_label not in {"yes", "no"}:
        return ""
    return "yes" if human_label == judge_label else "no"


def _match_count(labels: Sequence[str], judge_label: str) -> int:
    if judge_label not in {"yes", "no"}:
        return 0
    return sum(1 for label in labels if label == judge_label)


def _gold_fields(item: Mapping[str, Any]) -> Dict[str, str]:
    options = item.get("gold_options") or []
    if not isinstance(options, list):
        options = []
    gold_1 = str(options[0]) if options else str(item.get("gold") or "")
    gold_2 = str(options[1]) if len(options) > 1 else ""
    return {"gold_1": gold_1, "gold_2": gold_2}


def _analysis_item(
    item: Mapping[str, Any], human_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    ds_correct = _judge_correct(item, "ds-v3.2")
    gpt_correct = _judge_correct(item, "gpt-5.2")
    human_final = str(human_summary.get("human_final_correct") or "")
    human_labels = human_summary.get("human_labels", [])
    label_counts = human_summary.get("human_label_counts") or {}
    labels_by_annotator = human_summary.get("labels_by_annotator") or {}
    annotator_1_correct = str(labels_by_annotator.get(KNOWN_ANNOTATORS[0]) or "")
    annotator_2_correct = str(labels_by_annotator.get(KNOWN_ANNOTATORS[1]) or "")
    n_human_labels = int(human_summary.get("n_human_labels", 0))
    n_human_matches_ds = _match_count(human_labels, ds_correct)
    n_human_matches_gpt = _match_count(human_labels, gpt_correct)

    return {
        "audit_id": item.get("audit_id"),
        "human_status": human_summary.get("human_status"),
        "needs_adjudication": (
            "yes" if human_summary.get("needs_adjudication") else "no"
        ),
        "n_human_labels": n_human_labels,
        "annotator_1_correct": annotator_1_correct,
        "annotator_2_correct": annotator_2_correct,
        "human_consensus_correct": human_summary.get("human_consensus_correct"),
        "human_final_correct": human_final,
        "human_labels": human_labels,
        "annotator_ids": human_summary.get("annotator_ids", []),
        "n_human_yes": label_counts.get("yes", 0),
        "n_human_no": label_counts.get("no", 0),
        "ds_v3_2_correct": ds_correct,
        "gpt_5_2_correct": gpt_correct,
        "llm_judges_agree": _label_match(ds_correct, gpt_correct),
        "human_matches_ds_v3_2": _label_match(human_final, ds_correct),
        "human_matches_gpt_5_2": _label_match(human_final, gpt_correct),
        "n_human_matches_ds_v3_2": n_human_matches_ds,
        "n_human_matches_gpt_5_2": n_human_matches_gpt,
        "human_match_rate_ds_v3_2": (
            n_human_matches_ds / n_human_labels if n_human_labels else ""
        ),
        "human_match_rate_gpt_5_2": (
            n_human_matches_gpt / n_human_labels if n_human_labels else ""
        ),
        "annotator_1_matches_ds_v3_2": _label_match(annotator_1_correct, ds_correct),
        "annotator_1_matches_gpt_5_2": _label_match(annotator_1_correct, gpt_correct),
        "annotator_2_matches_ds_v3_2": _label_match(annotator_2_correct, ds_correct),
        "annotator_2_matches_gpt_5_2": _label_match(annotator_2_correct, gpt_correct),
        "model_short": item.get("model_short"),
        "condition": item.get("condition"),
        "seed": item.get("seed"),
        "primary_stratum": item.get("primary_stratum"),
        "sampling_weight_primary": item.get("sampling_weight_primary"),
        "input": item.get("input"),
        **_gold_fields(item),
        "prediction": item.get("prediction"),
        "source_file": item.get("source_file"),
        "item_id": item.get("item_id"),
    }


def _judge_agreement_summary(
    merged_items: Sequence[Mapping[str, Any]], match_field: str
) -> Dict[str, Any]:
    values = [
        item.get(match_field)
        for item in merged_items
        if item.get(match_field) in {"yes", "no"}
    ]
    matches = sum(1 for value in values if value == "yes")
    return {
        "n": len(values),
        "matches": matches,
        "agreement_rate": matches / len(values) if values else None,
    }


def _individual_judge_agreement_summary(
    merged_items: Sequence[Mapping[str, Any]], match_count_field: str
) -> Dict[str, Any]:
    n_labels = sum(int(item.get("n_human_labels") or 0) for item in merged_items)
    matches = sum(int(item.get(match_count_field) or 0) for item in merged_items)
    return {
        "n": n_labels,
        "matches": matches,
        "agreement_rate": matches / n_labels if n_labels else None,
    }


def merge_human_annotations(
    key_path: Path,
    annotation_paths: Sequence[Path],
    output_dir: Path,
    output_stem: str = "aaai_human_eval_merged",
) -> Dict[str, Any]:
    key_payload = load_json(key_path)
    key_items = key_payload.get("items") if isinstance(key_payload, dict) else None
    if not isinstance(key_items, list):
        raise ValueError("Key file must contain an items list.")

    annotations_by_id = load_human_annotations(annotation_paths)
    merged_items: List[Dict[str, Any]] = []
    for item in key_items:
        audit_id = str(item.get("audit_id") or "")
        annotations = annotations_by_id.get(audit_id, [])
        human_summary = _human_summary(annotations)
        merged_items.append(_analysis_item(item, human_summary))

    human_comparison_count = sum(
        1
        for item in merged_items
        if item["human_status"] in {"agreement", "disagreement"}
    )
    human_agreement_count = sum(
        1 for item in merged_items if item["human_status"] == "agreement"
    )

    summary = {
        "n_key_items": len(key_items),
        "n_items_with_annotations": sum(
            1 for item in merged_items if item["n_human_labels"] > 0
        ),
        "n_single_annotations": sum(
            1 for item in merged_items if item["human_status"] == "single_annotation"
        ),
        "n_items_needing_adjudication": sum(
            1 for item in merged_items if item["needs_adjudication"] == "yes"
        ),
        "n_human_agreements": human_agreement_count,
        "n_human_disagreements": sum(
            1 for item in merged_items if item["human_status"] == "disagreement"
        ),
        "n_items_with_final_label": sum(
            1
            for item in merged_items
            if item.get("human_final_correct") in {"yes", "no"}
        ),
        "human_status_counts": dict(
            Counter(item["human_status"] for item in merged_items)
        ),
        "human_human_agreement_rate": (
            human_agreement_count / human_comparison_count
            if human_comparison_count
            else None
        ),
        "llm_human_agreement": {
            "ds-v3.2": _judge_agreement_summary(merged_items, "human_matches_ds_v3_2"),
            "gpt-5.2": _judge_agreement_summary(merged_items, "human_matches_gpt_5_2"),
        },
        "llm_individual_human_agreement": {
            "ds-v3.2": _individual_judge_agreement_summary(
                merged_items, "n_human_matches_ds_v3_2"
            ),
            "gpt-5.2": _individual_judge_agreement_summary(
                merged_items, "n_human_matches_gpt_5_2"
            ),
        },
        "annotation_files": [str(path) for path in annotation_paths],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    jsonl_path = output_dir / f"{output_stem}.jsonl"
    csv_path = output_dir / f"{output_stem}.csv"

    save_json({"summary": summary, "items": merged_items}, json_path)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for item in merged_items:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MERGED_CSV_COLUMNS)
        writer.writeheader()
        for item in merged_items:
            writer.writerow(
                {
                    "audit_id": item.get("audit_id"),
                    "human_status": item.get("human_status"),
                    "needs_adjudication": item.get("needs_adjudication"),
                    "n_human_labels": item.get("n_human_labels"),
                    "annotator_1_correct": item.get("annotator_1_correct"),
                    "annotator_2_correct": item.get("annotator_2_correct"),
                    "human_consensus_correct": item.get("human_consensus_correct"),
                    "human_final_correct": item.get("human_final_correct"),
                    "human_labels": json.dumps(item.get("human_labels", [])),
                    "annotator_ids": json.dumps(item.get("annotator_ids", [])),
                    "n_human_yes": item.get("n_human_yes"),
                    "n_human_no": item.get("n_human_no"),
                    "ds_v3_2_correct": item.get("ds_v3_2_correct"),
                    "gpt_5_2_correct": item.get("gpt_5_2_correct"),
                    "llm_judges_agree": item.get("llm_judges_agree"),
                    "human_matches_ds_v3_2": item.get("human_matches_ds_v3_2"),
                    "human_matches_gpt_5_2": item.get("human_matches_gpt_5_2"),
                    "n_human_matches_ds_v3_2": item.get("n_human_matches_ds_v3_2"),
                    "n_human_matches_gpt_5_2": item.get("n_human_matches_gpt_5_2"),
                    "human_match_rate_ds_v3_2": item.get("human_match_rate_ds_v3_2"),
                    "human_match_rate_gpt_5_2": item.get("human_match_rate_gpt_5_2"),
                    "annotator_1_matches_ds_v3_2": item.get(
                        "annotator_1_matches_ds_v3_2"
                    ),
                    "annotator_1_matches_gpt_5_2": item.get(
                        "annotator_1_matches_gpt_5_2"
                    ),
                    "annotator_2_matches_ds_v3_2": item.get(
                        "annotator_2_matches_ds_v3_2"
                    ),
                    "annotator_2_matches_gpt_5_2": item.get(
                        "annotator_2_matches_gpt_5_2"
                    ),
                    "model_short": item.get("model_short"),
                    "condition": item.get("condition"),
                    "seed": item.get("seed"),
                    "primary_stratum": item.get("primary_stratum"),
                    "sampling_weight_primary": item.get("sampling_weight_primary"),
                    "input": item.get("input"),
                    "gold_1": item.get("gold_1"),
                    "gold_2": item.get("gold_2"),
                    "prediction": item.get("prediction"),
                    "source_file": item.get("source_file"),
                    "item_id": item.get("item_id"),
                }
            )

    return {
        "summary": summary,
        "files": {
            "json": str(json_path),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
    }
