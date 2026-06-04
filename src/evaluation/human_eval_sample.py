"""Create stratified human-evaluation samples from paired judge outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.sax.saxutils import escape

from ..infra.io import load_json, save_json
from .judge_agreement import create_item_key

DEFAULT_JUDGES: Tuple[str, str] = ("ds-v3.2", "gpt-5.2")
DEFAULT_ANNOTATORS: Tuple[str, str] = ("Francesco", "Marco")

DEFAULT_AAAI_QUOTAS: Dict[str, int] = {
    "disagree_ds_no_gpt_yes": 33,
    "disagree_ds_yes_gpt_no": 327,
    "llm_agree_yes": 120,
    "llm_agree_no": 120,
}

PRIMARY_STRATA = tuple(DEFAULT_AAAI_QUOTAS.keys())

ANNOTATION_COLUMNS = [
    "audit_id",
    "input",
    "gold",
    "gold_options",
    "prediction",
    "correct",
    "annotator_id",
]

ERROR_CATEGORIES = [
    "equivalent",
    "syntax_error",
    "wrong_agent_or_coalition",
    "wrong_temporal_operator",
    "wrong_temporal_scope",
    "wrong_polarity",
    "wrong_atomic_proposition",
    "missing_condition",
    "extra_condition",
    "other",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_sort_key(
    record: Mapping[str, Any], sampling_seed: int, namespace: str
) -> str:
    return _stable_hash(sampling_seed, namespace, record["population_key"])


def _normalize_decision(value: Any) -> str:
    decision = str(value or "no").strip().lower()
    return decision if decision in {"yes", "no"} else "no"


def _source_stem(path: Path) -> str:
    return path.stem.split("__judge-")[0]


def _gold_options(item: Mapping[str, Any]) -> List[str]:
    options = item.get("gold_options") or item.get("expected_options")
    if isinstance(options, list) and options:
        return [str(option) for option in options]
    gold = item.get("gold") or item.get("expected") or ""
    return [str(gold)] if gold else []


def formula_profile(gold_options: Sequence[str], prediction: str) -> Dict[str, Any]:
    """Summarize formula operators so the sample can be audited by complexity."""
    gold_text = " || ".join(gold_options)

    def count_token(text: str, token: str) -> int:
        return len(
            re.findall(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text)
        )

    temporal_counts = {
        "G": count_token(gold_text, "G"),
        "F": count_token(gold_text, "F"),
        "X": count_token(gold_text, "X"),
        "U": count_token(gold_text, "U"),
    }
    connective_counts = {
        "and": gold_text.count("&&"),
        "or": gold_text.count("||"),
        "implies": gold_text.count("->"),
        "not": gold_text.count("!"),
    }
    coalition_count = gold_text.count("<<")
    complexity_score = (
        sum(temporal_counts.values())
        + sum(connective_counts.values())
        + coalition_count
        + max(0, len(gold_options) - 1)
    )

    if complexity_score <= 3 and len(gold_text) <= 100:
        complexity_bin = "simple"
    elif complexity_score <= 8 and len(gold_text) <= 240:
        complexity_bin = "medium"
    else:
        complexity_bin = "complex"

    return {
        "gold_option_count": len(gold_options),
        "gold_length_chars": len(gold_text),
        "prediction_length_chars": len(prediction or ""),
        "coalition_count": coalition_count,
        "has_negated_ability": "!<<" in gold_text.replace(" ", ""),
        "temporal_counts": temporal_counts,
        "connective_counts": connective_counts,
        "complexity_score": complexity_score,
        "complexity_bin": complexity_bin,
    }


def load_paired_judge_population(
    eval_dir: Path,
    judges: Sequence[str] = DEFAULT_JUDGES,
) -> List[Dict[str, Any]]:
    """Load row-level judgments present for every requested judge."""
    records: Dict[str, Dict[str, Any]] = {}

    for judge_name in judges:
        judge_dir = eval_dir / judge_name
        if not judge_dir.exists():
            raise FileNotFoundError(f"Judge directory not found: {judge_dir}")

        for json_path in sorted(judge_dir.glob("*.json")):
            data = load_json(json_path)
            if not isinstance(data, dict):
                continue

            detailed_results = data.get("detailed_results")
            if not isinstance(detailed_results, list):
                continue

            source_stem = _source_stem(json_path)
            run_metadata = {
                "source_stem": source_stem,
                "source_file": data.get("source_file"),
                "run_id": data.get("run_id") or source_stem,
                "model": data.get("model"),
                "model_short": data.get("model_short"),
                "condition": data.get("condition"),
                "seed": data.get("seed"),
                "finetuned": data.get("finetuned"),
                "few_shot": data.get("few_shot"),
                "num_few_shot": data.get("num_few_shot"),
                "git_commit": data.get("git_commit"),
                "dataset_path": data.get("dataset_path"),
                "source_sha256": data.get("source_sha256"),
            }

            for item in detailed_results:
                if not isinstance(item, dict):
                    continue
                item_key = f"{source_stem}::{create_item_key(item)}"
                gold_options = _gold_options(item)
                prediction = str(item.get("prediction") or "")

                record = records.setdefault(
                    item_key,
                    {
                        "population_key": item_key,
                        "annotation_unit_key": _stable_hash(
                            item.get("input") or "",
                            gold_options,
                            prediction,
                        ),
                        "item_id": item.get("id"),
                        "input": item.get("input") or "",
                        "gold": item.get("gold")
                        or (gold_options[0] if gold_options else ""),
                        "gold_options": gold_options,
                        "prediction": prediction,
                        "judge_decisions": {},
                        **run_metadata,
                    },
                )

                record["judge_decisions"][judge_name] = {
                    "correct": _normalize_decision(item.get("correct")),
                    "reasoning": item.get("reasoning") or "",
                    "decision_method": item.get("decision_method") or "unknown",
                    "judge_parse_status": item.get("judge_parse_status"),
                    "judge_latency_ms": item.get("judge_latency_ms"),
                    "prompt_version": item.get("prompt_version"),
                    "judge_prompt_sha256": item.get("judge_prompt_sha256"),
                }

    paired_records = [
        record
        for record in records.values()
        if all(judge_name in record["judge_decisions"] for judge_name in judges)
    ]

    for record in paired_records:
        record["primary_stratum"] = decision_stratum(record, judges)
        record["source_stratum"] = source_stratum(record)
        record["formula_profile"] = formula_profile(
            record["gold_options"], record["prediction"]
        )

    return paired_records


def decision_stratum(
    record: Mapping[str, Any], judges: Sequence[str] = DEFAULT_JUDGES
) -> str:
    """Return the primary audit stratum for one aligned item."""
    judge_decisions = record["judge_decisions"]
    decision_methods = {
        judge_decisions[judge_name].get("decision_method") for judge_name in judges
    }
    if "exact" in decision_methods:
        return "exact_match"

    first_judge, second_judge = judges[0], judges[1]
    first_decision = judge_decisions[first_judge]["correct"]
    second_decision = judge_decisions[second_judge]["correct"]

    if first_decision == second_decision == "yes":
        return "llm_agree_yes"
    if first_decision == second_decision == "no":
        return "llm_agree_no"
    if first_decision == "yes" and second_decision == "no":
        return (
            f"disagree_{_judge_label(first_judge)}_yes_{_judge_label(second_judge)}_no"
        )
    if first_decision == "no" and second_decision == "yes":
        return (
            f"disagree_{_judge_label(first_judge)}_no_{_judge_label(second_judge)}_yes"
        )
    return "other"


def _judge_label(judge_name: str) -> str:
    if judge_name.startswith("ds-"):
        return "ds"
    if judge_name.startswith("gpt-"):
        return "gpt"
    return re.sub(r"[^A-Za-z0-9]+", "_", judge_name).strip("_").lower()


def source_stratum(record: Mapping[str, Any]) -> str:
    return "|".join(
        str(record.get(key)) for key in ("model_short", "condition", "seed")
    )


def _normalize_requested_quotas(
    requested_quotas: Mapping[str, int], population_counts: Mapping[str, int]
) -> Dict[str, int]:
    quotas = {
        stratum: min(max(0, int(quota)), population_counts.get(stratum, 0))
        for stratum, quota in requested_quotas.items()
    }
    requested_total = sum(max(0, int(quota)) for quota in requested_quotas.values())
    shortfall = requested_total - sum(quotas.values())
    if shortfall <= 0:
        return quotas

    fill_order = [
        "disagree_ds_yes_gpt_no",
        "llm_agree_yes",
        "llm_agree_no",
        "exact_match",
        "disagree_ds_no_gpt_yes",
    ]
    for stratum in fill_order:
        available = population_counts.get(stratum, 0) - quotas.get(stratum, 0)
        if available <= 0:
            continue
        added = min(shortfall, available)
        quotas[stratum] = quotas.get(stratum, 0) + added
        shortfall -= added
        if shortfall == 0:
            break
    return quotas


def _apportion_counts(group_sizes: Mapping[str, int], quota: int) -> Dict[str, int]:
    total_available = sum(group_sizes.values())
    quota = min(quota, total_available)
    if quota <= 0 or total_available <= 0:
        return {group: 0 for group in group_sizes}

    raw_shares = {
        group: quota * group_size / total_available
        for group, group_size in group_sizes.items()
    }
    allocations = {
        group: min(group_sizes[group], int(raw_share))
        for group, raw_share in raw_shares.items()
    }
    remaining = quota - sum(allocations.values())

    while remaining > 0:
        candidates = [
            group
            for group, group_size in group_sizes.items()
            if allocations[group] < group_size
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda group: (
                raw_shares[group] - int(raw_shares[group]),
                group_sizes[group] - allocations[group],
                group,
            ),
            reverse=True,
        )
        allocations[candidates[0]] += 1
        remaining -= 1

    return allocations


def _sample_bucket(
    candidates: Sequence[Dict[str, Any]],
    quota: int,
    sampling_seed: int,
    namespace: str,
) -> List[Dict[str, Any]]:
    grouped_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[candidate["source_stratum"]].append(candidate)

    group_sizes = {
        group: len(group_candidates)
        for group, group_candidates in grouped_candidates.items()
    }
    allocations = _apportion_counts(group_sizes, quota)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    for group, group_candidates in grouped_candidates.items():
        ordered_candidates = sorted(
            group_candidates,
            key=lambda record: _stable_sort_key(record, sampling_seed, namespace),
        )
        for record in ordered_candidates[: allocations.get(group, 0)]:
            selected.append(record)
            selected_keys.add(record["population_key"])

    if len(selected) < min(quota, len(candidates)):
        remaining_candidates = [
            candidate
            for candidate in candidates
            if candidate["population_key"] not in selected_keys
        ]
        remaining_candidates.sort(
            key=lambda record: _stable_sort_key(
                record, sampling_seed, f"{namespace}:fill"
            )
        )
        selected.extend(
            remaining_candidates[: min(quota, len(candidates)) - len(selected)]
        )

    return selected


def stratified_sample(
    population: Sequence[Dict[str, Any]],
    quotas: Mapping[str, int],
    sampling_seed: int,
) -> List[Dict[str, Any]]:
    population_counts = Counter(record["primary_stratum"] for record in population)
    resolved_quotas = _normalize_requested_quotas(quotas, population_counts)
    by_primary: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in population:
        by_primary[record["primary_stratum"]].append(record)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    selected_annotation_units = set()
    for stratum, quota in resolved_quotas.items():
        candidates = by_primary.get(stratum, [])
        bucket_selected = _sample_bucket(
            candidates,
            quota,
            sampling_seed,
            f"primary:{stratum}",
        )
        unique_bucket_selected: List[Dict[str, Any]] = []
        for record in bucket_selected:
            annotation_unit = record["annotation_unit_key"]
            if annotation_unit in selected_annotation_units:
                continue
            unique_bucket_selected.append(record)
            selected_keys.add(record["population_key"])
            selected_annotation_units.add(annotation_unit)

        if len(unique_bucket_selected) < min(quota, len(candidates)):
            fill_candidates = [
                candidate
                for candidate in candidates
                if candidate["population_key"] not in selected_keys
                and candidate["annotation_unit_key"] not in selected_annotation_units
            ]
            fill_candidates.sort(
                key=lambda record: _stable_sort_key(
                    record, sampling_seed, f"primary:{stratum}:unique-fill"
                )
            )
            needed = min(quota, len(candidates)) - len(unique_bucket_selected)
            for record in fill_candidates[:needed]:
                unique_bucket_selected.append(record)
                selected_keys.add(record["population_key"])
                selected_annotation_units.add(record["annotation_unit_key"])

        selected.extend(unique_bucket_selected)

    target_total = min(sum(resolved_quotas.values()), len(population))
    if len(selected) < target_total:
        fill_priority = {
            "disagree_ds_yes_gpt_no": 0,
            "llm_agree_yes": 1,
            "llm_agree_no": 2,
            "exact_match": 3,
            "disagree_ds_no_gpt_yes": 4,
        }
        fill_candidates = [
            candidate
            for candidate in population
            if candidate["population_key"] not in selected_keys
            and candidate["annotation_unit_key"] not in selected_annotation_units
        ]
        fill_candidates.sort(
            key=lambda record: (
                fill_priority.get(record["primary_stratum"], 99),
                _stable_sort_key(record, sampling_seed, "global-unique-fill"),
            )
        )
        for record in fill_candidates[: target_total - len(selected)]:
            selected.append(record)
            selected_keys.add(record["population_key"])
            selected_annotation_units.add(record["annotation_unit_key"])

    return selected


def _assign_audit_ids(
    records: Sequence[Dict[str, Any]], sampling_seed: int, prefix: str
) -> List[Dict[str, Any]]:
    ordered_records = sorted(
        records,
        key=lambda record: _stable_sort_key(record, sampling_seed, "presentation"),
    )
    assigned_records: List[Dict[str, Any]] = []
    for index, record in enumerate(ordered_records, start=1):
        copied_record = dict(record)
        copied_record["audit_id"] = f"{prefix}-{index:04d}"
        copied_record["presentation_order"] = index
        assigned_records.append(copied_record)
    return assigned_records


def _blind_item(record: Mapping[str, Any], annotator_id: str = "") -> Dict[str, Any]:
    return {
        "audit_id": record["audit_id"],
        "input": record["input"],
        "gold": record["gold"],
        "gold_options": record["gold_options"],
        "prediction": record["prediction"],
        "correct": "",
        "annotator_id": annotator_id,
    }


def _key_item(
    record: Mapping[str, Any],
    population_primary_counts: Mapping[str, int],
    sample_primary_counts: Mapping[str, int],
) -> Dict[str, Any]:
    primary_stratum = record["primary_stratum"]
    sample_count = max(1, sample_primary_counts.get(primary_stratum, 1))
    return {
        "audit_id": record["audit_id"],
        "presentation_order": record["presentation_order"],
        "population_key": record["population_key"],
        "annotation_unit_key": record.get("annotation_unit_key"),
        "primary_stratum": primary_stratum,
        "source_stratum": record["source_stratum"],
        "sampling_weight_primary": round(
            population_primary_counts.get(primary_stratum, 0) / sample_count, 6
        ),
        "source_stem": record.get("source_stem"),
        "source_file": record.get("source_file"),
        "run_id": record.get("run_id"),
        "item_id": record.get("item_id"),
        "model": record.get("model"),
        "model_short": record.get("model_short"),
        "condition": record.get("condition"),
        "seed": record.get("seed"),
        "finetuned": record.get("finetuned"),
        "few_shot": record.get("few_shot"),
        "num_few_shot": record.get("num_few_shot"),
        "git_commit": record.get("git_commit"),
        "dataset_path": record.get("dataset_path"),
        "source_sha256": record.get("source_sha256"),
        "formula_profile": record.get("formula_profile"),
        "input": record.get("input"),
        "gold": record.get("gold"),
        "gold_options": record.get("gold_options"),
        "prediction": record.get("prediction"),
        "judge_decisions": record.get("judge_decisions"),
    }


def _counter_dict(records: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(record.get(field)) for record in records).items()))


def _complexity_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(
        sorted(
            Counter(
                record.get("formula_profile", {}).get("complexity_bin", "unknown")
                for record in records
            ).items()
        )
    )


def _write_json_annotations(
    path: Path, metadata: Mapping[str, Any], records: Sequence[Dict[str, Any]]
) -> None:
    save_json(
        {
            "metadata": metadata,
            "annotation_fields": {
                "correct": "Use yes/no. This is the only required annotation field.",
            },
            "annotations": [_blind_item(record) for record in records],
        },
        path,
    )


def _write_csv_annotations(
    path: Path, records: Sequence[Dict[str, Any]], annotator_id: str = ""
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for record in records:
            item = _blind_item(record, annotator_id=annotator_id)
            item["gold_options"] = json.dumps(item["gold_options"], ensure_ascii=False)
            writer.writerow(item)


def _write_jsonl_annotations(
    path: Path, records: Sequence[Dict[str, Any]], annotator_id: str = ""
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(
                json.dumps(
                    _blind_item(record, annotator_id=annotator_id), ensure_ascii=False
                )
                + "\n"
            )


def _excel_column(index: int) -> str:
    column = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        column = chr(65 + remainder) + column
    return column


def _xlsx_cell(row_index: int, column_index: int, value: Any) -> str:
    cell_ref = f"{_excel_column(column_index)}{row_index}"
    text = "" if value is None else str(value)
    return (
        f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
        f"{escape(text)}"
        "</t></is></c>"
    )


def _write_xlsx_annotations(
    path: Path,
    records: Sequence[Dict[str, Any]],
    annotator_id: str = "",
    annotator_choices: Sequence[str] = DEFAULT_ANNOTATORS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[List[str]] = [ANNOTATION_COLUMNS]
    for record in records:
        item = _blind_item(record, annotator_id=annotator_id)
        item["gold_options"] = json.dumps(item["gold_options"], ensure_ascii=False)
        rows.append([str(item.get(column, "")) for column in ANNOTATION_COLUMNS])

    correct_column = _excel_column(ANNOTATION_COLUMNS.index("correct") + 1)
    annotator_column = _excel_column(ANNOTATION_COLUMNS.index("annotator_id") + 1)
    annotator_list = ",".join(annotator_choices)
    max_row = max(2, len(rows))
    dimension = f"A1:{_excel_column(len(ANNOTATION_COLUMNS))}{max_row}"
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(
            _xlsx_cell(row_index, column_index, value)
            for column_index, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_index}">{cells}</row>')

    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="14" customWidth="1"/>
    <col min="2" max="5" width="55" customWidth="1"/>
    <col min="6" max="7" width="16" customWidth="1"/>
  </cols>
    <sheetData>{''.join(sheet_rows)}</sheetData>
    <dataValidations count="2"><dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="{correct_column}2:{correct_column}{max_row}"><formula1>"yes,no"</formula1></dataValidation><dataValidation type="list" allowBlank="0" showErrorMessage="1" sqref="{annotator_column}2:{annotator_column}{max_row}"><formula1>"{escape(annotator_list)}"</formula1></dataValidation></dataValidations>
</worksheet>"""

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="annotations" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/styles.xml", styles_xml)


def _write_protocol(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary_strata = metadata["sample_by_primary_stratum"]
    protocol = f"""# Human Evaluation Protocol

This package is a blind, stratified audit set for calibrating the LLM judges used in the NL2ATL evaluation.

## Recommended Use

- Use the annotator-specific XLSX workbooks under `annotations/` for annotation.
- The `correct` column is restricted to a yes/no dropdown, and `annotator_id` is restricted to {"/".join(metadata.get("annotator_choices", []))}.
- Do not expose `aaai_human_eval_sample_key.json` to annotators until annotations are locked.
- Use two ATL-literate project annotators. They should annotate independently first, then resolve disagreements through a documented adjudication/discussion pass.
- Annotate all {metadata["sample_size"]} core sample items before analyzing judge agreement.
- If annotation budget allows, rerun with `--write_disagreement_pool` as a stricter follow-up that adjudicates every LLM-judge disagreement.
- Exact matches are excluded from the default human workload because they are accepted by deterministic normalization before LLM judging.

## Sampling Rationale

The sample is designed to calibrate the LLM judges on cases where deterministic exact match cannot decide correctness. It starts from the paired-judge population of {metadata["population_size"]} evaluated prediction items, each judged by DeepSeek V3.2 and GPT-5.2. Exact matches are excluded. The remaining items are grouped by judge-decision stratum, with disagreements oversampled because they are the most informative cases for identifying which judge is closer to human expert labels.

Agreement controls are retained: `llm_agree_yes` checks whether consensus acceptances are too permissive, and `llm_agree_no` checks whether consensus rejections are too strict. Duplicate `input` + `gold_options` + `prediction` triples are collapsed in the core sample to avoid redundant annotation, while the private key file preserves source metadata. Within each stratum, the sampler spreads examples across generator model, condition, and seed.

| Stratum | Meaning | Count |
|---|---|---:|
| `disagree_ds_no_gpt_yes` | DeepSeek rejects, GPT accepts | {primary_strata.get("disagree_ds_no_gpt_yes", 0)} |
| `disagree_ds_yes_gpt_no` | DeepSeek accepts, GPT rejects | {primary_strata.get("disagree_ds_yes_gpt_no", 0)} |
| `llm_agree_no` | both judges reject | {primary_strata.get("llm_agree_no", 0)} |
| `llm_agree_yes` | both judges accept | {primary_strata.get("llm_agree_yes", 0)} |
| **Total** |  | **{metadata["sample_size"]}** |

## Annotation Task

For each row, decide whether `prediction` is semantically equivalent to at least one formula in `gold_options` for the given natural-language `input`.

Mark `correct` as `yes` only when the prediction preserves the coalition/agent, temporal operator, temporal scope, polarity, logical structure, and atomic proposition meaning. Ignore whitespace and harmless parentheses. Do not mark a prediction correct when it changes an agent, changes temporal scope, flips polarity, drops or adds a condition, uses an unsupported alias, or introduces malformed ATL syntax.

Only the `correct` field is required. Use `yes` when the prediction is correct and `no` when it is incorrect. Add free-text notes only in a separate adjudication document if the two annotators disagree or a case needs discussion.

## What To Report In The Paper

- Human-human agreement before adjudication.
- The number of human-human disagreements before adjudication.
- Agreement of each LLM judge with the adjudicated human label.
- Accuracy of the LLM-judge consensus and each disagreement direction.
- A note that exact matches were accepted automatically by deterministic normalization and excluded from human annotation.
- Whether the main model ranking changes after replacing sampled LLM-judge labels with human adjudication.
- The sampling seed, quotas, and stratum counts from `aaai_human_eval_sample_metadata.json`.

## Sampling Summary

- Population size: {metadata["population_size"]}
- Core sample size: {metadata["sample_size"]}
- Sampling seed: {metadata["sampling_seed"]}
- Primary strata: {json.dumps(metadata["sample_by_primary_stratum"], sort_keys=True)}
"""
    path.write_text(protocol, encoding="utf-8")


def _package_sample(
    records: Sequence[Dict[str, Any]],
    population: Sequence[Dict[str, Any]],
    output_dir: Path,
    stem: str,
    metadata: Dict[str, Any],
    write_legacy_formats: bool = False,
    annotator_choices: Sequence[str] = DEFAULT_ANNOTATORS,
) -> Dict[str, str]:
    population_primary_counts = Counter(
        record["primary_stratum"] for record in population
    )
    sample_primary_counts = Counter(record["primary_stratum"] for record in records)
    key_records = [
        _key_item(record, population_primary_counts, sample_primary_counts)
        for record in records
    ]

    xlsx_path = output_dir / f"{stem}_blind.xlsx"
    key_path = output_dir / f"{stem}_key.json"

    files = {
        "blind_xlsx": str(xlsx_path),
        "key_json": str(key_path),
    }

    _write_xlsx_annotations(xlsx_path, records, annotator_choices=annotator_choices)

    if write_legacy_formats:
        json_path = output_dir / f"{stem}_blind.json"
        csv_path = output_dir / f"{stem}_blind.csv"
        jsonl_path = output_dir / f"{stem}_blind.jsonl"
        _write_json_annotations(json_path, metadata, records)
        _write_csv_annotations(csv_path, records)
        _write_jsonl_annotations(jsonl_path, records)
        files.update(
            {
                "blind_json": str(json_path),
                "blind_csv": str(csv_path),
                "blind_jsonl": str(jsonl_path),
            }
        )

    save_json({"metadata": metadata, "items": key_records}, key_path)

    return files


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "annotator"


def _write_annotator_workbooks(
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
    annotator_choices: Sequence[str],
    backup_existing: bool = False,
) -> Dict[str, str]:
    annotation_dir = output_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for annotator_id in annotator_choices:
        path = annotation_dir / f"{_safe_filename(annotator_id)}_blind.xlsx"
        if backup_existing and path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            backup_path = path.with_name(f"{path.stem}.{timestamp}.bak{path.suffix}")
            path.replace(backup_path)
        _write_xlsx_annotations(
            path,
            records,
            annotator_id=annotator_id,
            annotator_choices=annotator_choices,
        )
        files[annotator_id] = str(path)
    return files


def regenerate_annotator_workbooks_from_key(
    key_path: Path = Path(
        "outputs/LLM-evaluation/human_evaluation/aaai_human_eval_sample_key.json"
    ),
    output_dir: Path | None = None,
    annotator_choices: Sequence[str] = DEFAULT_ANNOTATORS,
    backup_existing: bool = True,
) -> Dict[str, str]:
    """Regenerate blank annotator workbooks from an existing private key."""
    key_payload = load_json(key_path)
    key_items = key_payload.get("items") if isinstance(key_payload, dict) else None
    if not isinstance(key_items, list):
        raise ValueError("Key file must contain an items list.")

    records = sorted(
        (item for item in key_items if isinstance(item, dict)),
        key=lambda item: int(item.get("presentation_order") or 0),
    )
    target_dir = output_dir or key_path.parent
    return _write_annotator_workbooks(
        records,
        target_dir,
        annotator_choices,
        backup_existing=backup_existing,
    )


def build_human_eval_sample(
    eval_dir: Path = Path("outputs/LLM-evaluation/evaluated_datasets"),
    output_dir: Path = Path("outputs/LLM-evaluation/human_evaluation"),
    judges: Sequence[str] = DEFAULT_JUDGES,
    quotas: Mapping[str, int] = DEFAULT_AAAI_QUOTAS,
    sampling_seed: int = 20260604,
    write_disagreement_pool: bool = False,
    write_legacy_formats: bool = False,
    annotator_choices: Sequence[str] = DEFAULT_ANNOTATORS,
) -> Dict[str, Any]:
    """Build blind annotation files, analysis keys, and metadata."""
    population = load_paired_judge_population(eval_dir, judges)
    if not population:
        raise ValueError(f"No paired judge records found in {eval_dir}")

    population_counts = Counter(record["primary_stratum"] for record in population)
    resolved_quotas = _normalize_requested_quotas(quotas, population_counts)
    sampled_records = stratified_sample(population, resolved_quotas, sampling_seed)
    sampled_records = _assign_audit_ids(sampled_records, sampling_seed, "HEVAL")

    sample_metadata: Dict[str, Any] = {
        "created_at": _utc_now(),
        "purpose": "AAAI-ready human audit sample for calibrating LLM-as-judge evaluation.",
        "eval_dir": str(eval_dir),
        "judges": list(judges),
        "sampling_seed": sampling_seed,
        "population_size": len(population),
        "sample_size": len(sampled_records),
        "requested_quotas": dict(quotas),
        "resolved_quotas": resolved_quotas,
        "annotator_choices": list(annotator_choices),
        "population_by_primary_stratum": dict(sorted(population_counts.items())),
        "sample_by_primary_stratum": dict(
            sorted(
                Counter(record["primary_stratum"] for record in sampled_records).items()
            )
        ),
        "sample_by_source_stratum": _counter_dict(sampled_records, "source_stratum"),
        "sample_by_complexity_bin": _complexity_counts(sampled_records),
        "sample_unique_annotation_units": len(
            {record["annotation_unit_key"] for record in sampled_records}
        ),
        "unique_annotation_policy": "The core sample avoids duplicate input/gold/prediction triples; any unique-unit shortfall in a quota is filled from other high-value strata.",
        "exact_match_policy": "Exact matches are excluded from the default human audit because they are accepted by the deterministic evaluator before LLM judging.",
        "blind_annotation_files_do_not_include": [
            "generator model identity",
            "judge model identity",
            "judge decisions",
            "judge reasoning",
            "primary sampling stratum",
        ],
        "recommended_reporting": [
            "human-human agreement",
            "LLM-human agreement per judge",
            "adjudicated accuracy by primary stratum",
            "deterministic exact-match policy",
            "model-ranking stability after human adjudication",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_files = _package_sample(
        sampled_records,
        population,
        output_dir,
        "aaai_human_eval_sample",
        sample_metadata,
        write_legacy_formats=write_legacy_formats,
        annotator_choices=annotator_choices,
    )
    annotator_files = _write_annotator_workbooks(
        sampled_records,
        output_dir,
        annotator_choices,
    )

    disagreement_files: Dict[str, str] = {}
    if write_disagreement_pool:
        disagreement_records = [
            record
            for record in population
            if str(record["primary_stratum"]).startswith("disagree_")
        ]
        disagreement_records = _assign_audit_ids(
            disagreement_records, sampling_seed, "DISAGREE"
        )
        disagreement_metadata = {
            **sample_metadata,
            "purpose": "Optional full pool of all LLM-judge disagreements for complete human adjudication.",
            "sample_size": len(disagreement_records),
            "sample_by_primary_stratum": dict(
                sorted(
                    Counter(
                        record["primary_stratum"] for record in disagreement_records
                    ).items()
                )
            ),
            "sample_by_source_stratum": _counter_dict(
                disagreement_records, "source_stratum"
            ),
            "sample_by_complexity_bin": _complexity_counts(disagreement_records),
        }
        disagreement_files = _package_sample(
            disagreement_records,
            population,
            output_dir,
            "aaai_disagreement_pool",
            disagreement_metadata,
            write_legacy_formats=write_legacy_formats,
            annotator_choices=annotator_choices,
        )

    metadata = {
        **sample_metadata,
        "files": {
            "core_sample": sample_files,
            "annotator_workbooks": annotator_files,
            "disagreement_pool": disagreement_files,
            "protocol": str(output_dir / "human_evaluation_protocol.md"),
            "metadata": str(output_dir / "aaai_human_eval_sample_metadata.json"),
        },
    }
    save_json(metadata, output_dir / "aaai_human_eval_sample_metadata.json")
    _write_protocol(output_dir / "human_evaluation_protocol.md", metadata)

    return metadata
