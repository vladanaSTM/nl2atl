"""Merge blind human annotations with the private human-eval key."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from ..infra.io import load_json, save_json
from ..infra.xlsx import XlsxDropdown, write_xlsx_sheet
from .judge_agreement import (
    _interpret_kappa,
    compute_cohen_kappa,
    compute_fleiss_kappa,
    compute_krippendorff_alpha,
)

# Annotator ids that represent a post-hoc adjudication/consensus pass rather
# than an independent annotator. Their labels become the final gold label but are
# excluded from inter-annotator (human-human) reliability.
ADJUDICATION_ANNOTATOR_IDS = frozenset(
    {"adjudicated", "adjudicator", "consensus", "final"}
)

DEFAULT_JUDGES: tuple[str, ...] = ("ds-v3.2", "gpt-5.2")
DEFAULT_ANNOTATORS: tuple[str, ...] = ("annotator_1", "annotator_2")

_PENDING_LABELS = frozenset(
    {
        "unannotated",
        "single_annotation",
        "pending_second_annotation",
        "no_consensus",
        "pending_adjudication",
    }
)


def _judge_slug(judge_name: str) -> str:
    """Turn a judge id like ``ds-v3.2`` into a column-safe slug ``ds_v3_2``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(judge_name)).strip("_").lower()


def _annotator_slug(annotator_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(annotator_id)).strip("_").lower()


def build_merged_csv_columns(
    annotator_ids: Sequence[str],
    judge_names: Sequence[str],
) -> List[str]:
    """Build the merged-CSV header for an arbitrary number of annotators/judges."""
    judge_slugs = [_judge_slug(judge) for judge in judge_names]
    annotator_slugs = [_annotator_slug(annotator) for annotator in annotator_ids]

    columns: List[str] = [
        "audit_id",
        "human_status",
        "needs_adjudication",
        "n_human_labels",
    ]
    columns += [f"{slug}_correct" for slug in annotator_slugs]
    columns += [f"{slug}_notes" for slug in annotator_slugs]
    columns += [
        "human_consensus_correct",
        "human_final_correct",
        "adjudication_notes",
        "human_labels",
        "annotator_ids",
        "n_human_yes",
        "n_human_no",
    ]
    columns += [f"{slug}_correct" for slug in judge_slugs]
    columns += ["llm_judges_agree"]
    columns += [f"human_matches_{slug}" for slug in judge_slugs]
    columns += [f"n_human_matches_{slug}" for slug in judge_slugs]
    columns += [f"human_match_rate_{slug}" for slug in judge_slugs]
    for annotator_slug in annotator_slugs:
        columns += [f"{annotator_slug}_matches_{slug}" for slug in judge_slugs]
    columns += [
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
    return columns


# Backward-compatible default header (two annotators, the two project judges).
MERGED_CSV_COLUMNS = build_merged_csv_columns(DEFAULT_ANNOTATORS, DEFAULT_JUDGES)


def _normalize_correct(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    value_str = str(value or "").strip().lower()
    if value_str in {"yes", "y", "true", "1", "correct"}:
        return "yes"
    if value_str in {"no", "n", "false", "0", "incorrect"}:
        return "no"
    return ""


_NOTES_FIELD_CANDIDATES = frozenset(
    {"notes", "note", "comment", "comments", "reasoning", "rationale", "deliberation"}
)


def _extract_notes(row: Mapping[str, Any]) -> str:
    """Return free-text notes from any notes-like column (case-insensitive).

    Annotators may hand-add the column under a few common headers; we accept
    any of them so existing workbooks do not need to be regenerated.
    """
    for key, value in row.items():
        if str(key).strip().lower() in _NOTES_FIELD_CANDIDATES:
            text = str(value or "").strip()
            if text:
                return text
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
                "notes": _extract_notes(row),
            }
            annotations_by_id.setdefault(audit_id, []).append(normalized_row)
    return annotations_by_id


def _human_summary(annotations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final_annotations = [
        annotation
        for annotation in annotations
        if str(annotation.get("annotator_id") or "").strip().lower()
        in ADJUDICATION_ANNOTATOR_IDS
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
    notes_by_annotator = {
        str(annotation.get("annotator_id") or ""): str(
            annotation.get("notes") or ""
        ).strip()
        for annotation in independent_annotations
        if str(annotation.get("notes") or "").strip()
    }
    adjudication_notes = " | ".join(
        str(annotation.get("notes") or "").strip()
        for annotation in final_annotations
        if str(annotation.get("notes") or "").strip()
    )
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
        "notes_by_annotator": notes_by_annotator,
        "adjudication_notes": adjudication_notes,
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
    if human_label in _PENDING_LABELS:
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


def _all_judges_agree(judge_labels: Sequence[str]) -> str:
    """Return yes/no when every judge produced a binary label and they match."""
    binary = [label for label in judge_labels if label in {"yes", "no"}]
    if len(binary) < 2 or len(binary) != len(judge_labels):
        return ""
    return "yes" if len(set(binary)) == 1 else "no"


def _analysis_item(
    item: Mapping[str, Any],
    human_summary: Mapping[str, Any],
    annotator_ids: Sequence[str],
    judge_names: Sequence[str],
) -> Dict[str, Any]:
    human_final = str(human_summary.get("human_final_correct") or "")
    human_labels = human_summary.get("human_labels", [])
    label_counts = human_summary.get("human_label_counts") or {}
    labels_by_annotator = human_summary.get("labels_by_annotator") or {}
    n_human_labels = int(human_summary.get("n_human_labels", 0))

    judge_correct = {judge: _judge_correct(item, judge) for judge in judge_names}

    record: Dict[str, Any] = {
        "audit_id": item.get("audit_id"),
        "human_status": human_summary.get("human_status"),
        "needs_adjudication": (
            "yes" if human_summary.get("needs_adjudication") else "no"
        ),
        "n_human_labels": n_human_labels,
    }

    annotator_correct = {
        annotator: str(labels_by_annotator.get(annotator) or "")
        for annotator in annotator_ids
    }
    notes_by_annotator = human_summary.get("notes_by_annotator") or {}
    for annotator in annotator_ids:
        record[f"{_annotator_slug(annotator)}_correct"] = annotator_correct[annotator]
    for annotator in annotator_ids:
        record[f"{_annotator_slug(annotator)}_notes"] = str(
            notes_by_annotator.get(annotator) or ""
        )

    record.update(
        {
            "human_consensus_correct": human_summary.get("human_consensus_correct"),
            "human_final_correct": human_final,
            "adjudication_notes": str(human_summary.get("adjudication_notes") or ""),
            "human_labels": human_labels,
            "annotator_ids": human_summary.get("annotator_ids", []),
            "n_human_yes": label_counts.get("yes", 0),
            "n_human_no": label_counts.get("no", 0),
        }
    )

    for judge in judge_names:
        record[f"{_judge_slug(judge)}_correct"] = judge_correct[judge]

    record["llm_judges_agree"] = _all_judges_agree(
        [judge_correct[judge] for judge in judge_names]
    )

    for judge in judge_names:
        slug = _judge_slug(judge)
        record[f"human_matches_{slug}"] = _label_match(human_final, judge_correct[judge])
    for judge in judge_names:
        slug = _judge_slug(judge)
        record[f"n_human_matches_{slug}"] = _match_count(
            human_labels, judge_correct[judge]
        )
    for judge in judge_names:
        slug = _judge_slug(judge)
        matches = _match_count(human_labels, judge_correct[judge])
        record[f"human_match_rate_{slug}"] = (
            matches / n_human_labels if n_human_labels else ""
        )

    for annotator in annotator_ids:
        annotator_slug = _annotator_slug(annotator)
        for judge in judge_names:
            record[f"{annotator_slug}_matches_{_judge_slug(judge)}"] = _label_match(
                annotator_correct[annotator], judge_correct[judge]
            )

    record.update(
        {
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
    )
    return record


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


def _human_human_reliability(
    aligned_humans: Mapping[str, Mapping[str, str]],
    annotator_ids: Sequence[str],
) -> Dict[str, Any]:
    """Chance-corrected inter-annotator reliability over independent labels.

    Cohen's kappa is reported only for the two-annotator case (where it is
    defined); Fleiss' kappa and Krippendorff's alpha generalize to any number
    of annotators and tolerate items that were not labeled by everyone.
    """
    multi = {
        audit_id: labels
        for audit_id, labels in aligned_humans.items()
        if len(labels) >= 2
    }
    result: Dict[str, Any] = {
        "annotators": list(annotator_ids),
        "n_annotators": len(annotator_ids),
        "n_items_multi_annotator": len(multi),
        "cohen_kappa": None,
        "cohen_kappa_interpretation": None,
        "fleiss_kappa": None,
        "fleiss_kappa_interpretation": None,
        "krippendorff_alpha": None,
        "krippendorff_alpha_interpretation": None,
    }
    if not multi:
        return result

    if len(annotator_ids) == 2:
        first, second = annotator_ids
        paired = [
            labels
            for labels in multi.values()
            if first in labels and second in labels
        ]
        if paired:
            labels_first = [labels[first] for labels in paired]
            labels_second = [labels[second] for labels in paired]
            kappa = compute_cohen_kappa(labels_first, labels_second)
            result["cohen_kappa"] = round(kappa, 4)
            result["cohen_kappa_interpretation"] = _interpret_kappa(kappa)

    counts_matrix = np.array(
        [
            [
                sum(1 for label in labels.values() if label == "yes"),
                sum(1 for label in labels.values() if label == "no"),
            ]
            for labels in multi.values()
        ],
        dtype=float,
    )
    fleiss = compute_fleiss_kappa(counts_matrix)
    result["fleiss_kappa"] = round(float(fleiss), 4)
    result["fleiss_kappa_interpretation"] = _interpret_kappa(fleiss)

    alpha = compute_krippendorff_alpha(
        {audit_id: dict(labels) for audit_id, labels in multi.items()},
        list(annotator_ids),
    )
    alpha_value = alpha.get("alpha")
    if alpha_value is not None:
        result["krippendorff_alpha"] = round(float(alpha_value), 4)
        result["krippendorff_alpha_interpretation"] = _interpret_kappa(alpha_value)
    return result


def _llm_human_reliability(
    merged_items: Sequence[Mapping[str, Any]],
    judge_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Cohen's kappa + raw accuracy of each judge against the human gold label."""
    result: Dict[str, Dict[str, Any]] = {}
    for judge in judge_names:
        slug = _judge_slug(judge)
        pairs = [
            (item.get("human_final_correct"), item.get(f"{slug}_correct"))
            for item in merged_items
            if item.get("human_final_correct") in {"yes", "no"}
            and item.get(f"{slug}_correct") in {"yes", "no"}
        ]
        if not pairs:
            result[judge] = {
                "n": 0,
                "accuracy": None,
                "cohen_kappa": None,
                "cohen_kappa_interpretation": None,
            }
            continue
        human_labels = [str(human) for human, _ in pairs]
        judge_labels = [str(judge_label) for _, judge_label in pairs]
        matches = sum(1 for human, judge_label in pairs if human == judge_label)
        kappa = compute_cohen_kappa(judge_labels, human_labels)
        result[judge] = {
            "n": len(pairs),
            "accuracy": round(matches / len(pairs), 4),
            "cohen_kappa": round(kappa, 4),
            "cohen_kappa_interpretation": _interpret_kappa(kappa),
        }
    return result


def _discover_judge_names(key_items: Sequence[Mapping[str, Any]]) -> List[str]:
    judge_name_set: set[str] = set()
    for item in key_items:
        decisions = item.get("judge_decisions")
        if isinstance(decisions, Mapping):
            judge_name_set.update(str(name) for name in decisions.keys())
    return sorted(judge_name_set) or list(DEFAULT_JUDGES)


def _discover_annotator_ids(
    annotations_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[str]:
    annotator_set: set[str] = set()
    for annotations in annotations_by_id.values():
        for annotation in annotations:
            annotator_id = str(annotation.get("annotator_id") or "").strip()
            if not annotator_id:
                continue
            if annotator_id.lower() in ADJUDICATION_ANNOTATOR_IDS:
                continue
            annotator_set.add(annotator_id)
    return sorted(annotator_set)


def _write_human_verdict_adapter(
    key_items: Sequence[Mapping[str, Any]],
    human_summaries: Sequence[Mapping[str, Any]],
    key_path: Path,
    adapter_path: Path,
) -> int:
    """Write a per-item human-gold file consumable by the agreement pipeline.

    Each record keeps the original ``source_file``/``gold`` so that the human
    verdict aligns with the LLM judges on the exact same item key.
    """
    adapter_items = []
    for key_item, human_summary in zip(key_items, human_summaries):
        if human_summary.get("human_final_correct") not in {"yes", "no"}:
            continue
        adapter_item = {
            "audit_id": key_item.get("audit_id"),
            "source_file": key_item.get("source_file"),
            "input": key_item.get("input"),
            "gold": key_item.get("gold"),
            "prediction": key_item.get("prediction"),
            "correct": human_summary.get("human_final_correct"),
        }
        adjudication_notes = str(human_summary.get("adjudication_notes") or "").strip()
        if adjudication_notes:
            adapter_item["adjudication_notes"] = adjudication_notes
        adapter_items.append(adapter_item)
    payload = {
        "human_label": "human",
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_key": str(key_path),
        "n_items": len(adapter_items),
        "items": adapter_items,
    }
    save_json(payload, adapter_path)
    return len(adapter_items)


ADJUDICATION_WORKBOOK_FILL_COLUMNS: tuple[str, ...] = ("correct", "notes")


def _adjudication_columns(annotator_ids: Sequence[str]) -> List[str]:
    """Columns for the adjudication workbook: context first, then fill fields."""
    columns = ["audit_id", "input", "gold_1", "gold_2", "prediction"]
    columns += [f"{_annotator_slug(annotator)}_correct" for annotator in annotator_ids]
    columns += [f"{_annotator_slug(annotator)}_notes" for annotator in annotator_ids]
    columns += ["correct", "annotator_id", "notes"]
    return columns


def write_adjudication_workbook(
    merged_items: Sequence[Mapping[str, Any]],
    output_path: Path,
    annotator_ids: Sequence[str],
    adjudicator_id: str = "adjudicated",
) -> int:
    """Write an XLSX of the items the annotators disagree on, for deliberation.

    Each row shows the shared context plus every annotator's verdict and note so
    the annotators can deliberate. They fill the blank ``correct`` (yes/no) and
    ``notes`` columns; ``annotator_id`` is pre-set to ``adjudicator_id`` so that
    re-running the merge with this file applies the agreed label as the final
    human gold (and keeps the rationale in ``adjudication_notes``).
    """
    columns = _adjudication_columns(annotator_ids)
    rows: List[List[str]] = []
    for item in merged_items:
        if item.get("needs_adjudication") != "yes":
            continue
        row: List[str] = []
        for column in columns:
            if column == "annotator_id":
                row.append(adjudicator_id)
            elif column in ADJUDICATION_WORKBOOK_FILL_COLUMNS:
                row.append("")
            else:
                row.append(str(item.get(column, "") or ""))
        rows.append(row)

    column_widths: Dict[str, float] = {
        "audit_id": 14,
        "input": 55,
        "gold_1": 45,
        "gold_2": 45,
        "prediction": 45,
        "correct": 12,
        "annotator_id": 14,
        "notes": 50,
    }
    for annotator in annotator_ids:
        column_widths[f"{_annotator_slug(annotator)}_correct"] = 16
        column_widths[f"{_annotator_slug(annotator)}_notes"] = 40

    write_xlsx_sheet(
        output_path,
        columns,
        rows,
        dropdowns=[
            XlsxDropdown("correct", ("yes", "no"), allow_blank=True),
            XlsxDropdown("annotator_id", (adjudicator_id,), allow_blank=False),
        ],
        column_widths=column_widths,
        sheet_name="adjudication",
    )
    return len(rows)


def merge_human_annotations(
    key_path: Path,
    annotation_paths: Sequence[Path],
    output_dir: Path,
    output_stem: str = "human_eval_merged",
    refresh_adjudication: bool = False,
) -> Dict[str, Any]:
    key_payload = load_json(key_path)
    key_items = key_payload.get("items") if isinstance(key_payload, dict) else None
    if not isinstance(key_items, list):
        raise ValueError("Key file must contain an items list.")

    annotations_by_id = load_human_annotations(annotation_paths)
    judge_names = _discover_judge_names(key_items)
    annotator_ids = _discover_annotator_ids(annotations_by_id)
    csv_columns = build_merged_csv_columns(annotator_ids, judge_names)

    merged_items: List[Dict[str, Any]] = []
    human_summaries: List[Dict[str, Any]] = []
    aligned_humans: Dict[str, Dict[str, str]] = {}
    for item in key_items:
        audit_id = str(item.get("audit_id") or "")
        annotations = annotations_by_id.get(audit_id, [])
        human_summary = _human_summary(annotations)
        human_summaries.append(human_summary)
        merged_items.append(
            _analysis_item(item, human_summary, annotator_ids, judge_names)
        )
        labels_by_annotator = human_summary.get("labels_by_annotator") or {}
        if labels_by_annotator:
            aligned_humans[audit_id] = dict(labels_by_annotator)

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
        "judges": judge_names,
        "annotators": annotator_ids,
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
        "human_human_reliability": _human_human_reliability(
            aligned_humans, annotator_ids
        ),
        "llm_human_agreement": {
            judge: _judge_agreement_summary(
                merged_items, f"human_matches_{_judge_slug(judge)}"
            )
            for judge in judge_names
        },
        "llm_individual_human_agreement": {
            judge: _individual_judge_agreement_summary(
                merged_items, f"n_human_matches_{_judge_slug(judge)}"
            )
            for judge in judge_names
        },
        "llm_human_reliability": _llm_human_reliability(merged_items, judge_names),
        "annotation_files": [str(path) for path in annotation_paths],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    jsonl_path = output_dir / f"{output_stem}.jsonl"
    csv_path = output_dir / f"{output_stem}.csv"
    adapter_path = output_dir / f"{output_stem}_adjudicated.json"

    n_adjudicated = _write_human_verdict_adapter(
        key_items, human_summaries, key_path, adapter_path
    )

    save_json({"summary": summary, "items": merged_items}, json_path)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for item in merged_items:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for item in merged_items:
            row = {}
            for column in csv_columns:
                value = item.get(column)
                if column in {"human_labels", "annotator_ids"}:
                    value = json.dumps(value or [])
                row[column] = value
            writer.writerow(row)

    # Auto-emit the disagreement workbook so the annotators can deliberate.
    # It is (re)written only while disagreements remain, and never overwrites an
    # in-progress fill of the same file unless refresh_adjudication is set.
    adjudication_path = output_dir / f"{output_stem}_adjudication.xlsx"
    n_needing = summary["n_items_needing_adjudication"]
    if n_needing > 0 and (refresh_adjudication or not adjudication_path.exists()):
        write_adjudication_workbook(merged_items, adjudication_path, annotator_ids)

    files = {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "adjudicated_human_gold": str(adapter_path),
    }
    if adjudication_path.exists():
        files["adjudication_workbook"] = str(adjudication_path)

    return {
        "summary": summary,
        "files": files,
        "n_adjudicated_human_labels": n_adjudicated,
    }
