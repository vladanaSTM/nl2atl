#!/usr/bin/env python3
"""Validate the canonical NL2ATL gold dataset.

The checker is intentionally dependency-free so it can run in CI and by
reviewers without installing the training stack. It validates the JSON schema,
ID discipline, non-empty gold outputs, light ATL/ATL* syntax, duplicate rows,
and optional alignment with the informal reviewer PDF exported from Drive.

Examples:
    python validate_dataset.py data/dataset_gold.json
    python validate_dataset.py data/dataset_gold.json --review-pdf docs/dataset_NL2ATL_review.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ID_RE = re.compile(r"^ex(\d{2,4})$")
COALITION_RE = re.compile(r"<<[^<>]+>>")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TEMPORAL_UNARY = {"X", "F", "G"}
BINARY_OPS = {"->", "&&", "||", "U"}


@dataclass
class Token:
    kind: str
    value: str
    pos: int


class FormulaSyntaxError(ValueError):
    pass


class FormulaParser:
    """Small permissive parser for the ATL/ATL* subset used in the dataset."""

    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.tokens = self._tokenize(formula)
        self.pos = 0

    def _tokenize(self, formula: str) -> list[Token]:
        tokens: list[Token] = []
        i = 0
        while i < len(formula):
            c = formula[i]
            if c.isspace():
                i += 1
                continue
            if formula.startswith("<<", i):
                end = formula.find(">>", i + 2)
                if end == -1:
                    raise FormulaSyntaxError(f"unterminated coalition at char {i}")
                value = formula[i : end + 2]
                inside = value[2:-2].strip()
                if not inside:
                    raise FormulaSyntaxError(f"empty coalition at char {i}")
                for member in inside.split(","):
                    member = member.strip()
                    if not IDENT_RE.fullmatch(member):
                        raise FormulaSyntaxError(
                            f"invalid coalition member {member!r} at char {i}"
                        )
                tokens.append(Token("COALITION", value, i))
                i = end + 2
                continue
            two = formula[i : i + 2]
            if two in {"->", "&&", "||"}:
                tokens.append(Token(two, two, i))
                i += 2
                continue
            if c in "()!,":
                tokens.append(Token(c, c, i))
                i += 1
                continue
            match = IDENT_RE.match(formula, i)
            if match:
                word = match.group(0)
                # Some gold formulas use compact temporal strings such as XF p.
                # Split strings made only of temporal unary operators into X F.
                if len(word) > 1 and all(ch in TEMPORAL_UNARY for ch in word):
                    for offset, ch in enumerate(word):
                        tokens.append(Token(ch, ch, i + offset))
                elif word in TEMPORAL_UNARY or word == "U":
                    tokens.append(Token(word, word, i))
                else:
                    tokens.append(Token("IDENT", word, i))
                i = match.end()
                continue
            raise FormulaSyntaxError(f"unexpected character {c!r} at char {i}")
        tokens.append(Token("EOF", "", len(formula)))
        return tokens

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def accept(self, kind: str) -> bool:
        if self.peek().kind == kind:
            self.pos += 1
            return True
        return False

    def expect(self, kind: str) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise FormulaSyntaxError(
                f"expected {kind}, found {tok.value or tok.kind!r} at char {tok.pos}"
            )
        self.pos += 1
        return tok

    def parse(self) -> None:
        self.parse_implication()
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise FormulaSyntaxError(f"trailing token {tok.value!r} at char {tok.pos}")

    def parse_implication(self) -> None:
        self.parse_or()
        if self.accept("->"):
            self.parse_implication()

    def parse_or(self) -> None:
        self.parse_and()
        while self.accept("||"):
            self.parse_and()

    def parse_and(self) -> None:
        self.parse_until()
        while self.accept("&&"):
            self.parse_until()

    def parse_until(self) -> None:
        self.parse_unary()
        while self.accept("U"):
            self.parse_unary()

    def parse_unary(self) -> None:
        if self.accept("!"):
            self.parse_unary()
            return
        if self.peek().kind in TEMPORAL_UNARY:
            self.pos += 1
            self.parse_unary()
            return
        if self.accept("COALITION"):
            self.parse_unary()
            return
        self.parse_primary()

    def parse_primary(self) -> None:
        if self.accept("IDENT"):
            return
        if self.accept("("):
            self.parse_implication()
            self.expect(")")
            return
        tok = self.peek()
        raise FormulaSyntaxError(f"expected atom or '(', found {tok.value!r} at char {tok.pos}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula).replace("“", '"').replace("”", '"')


def item_outputs(item: dict[str, Any]) -> list[str]:
    outputs = item.get("outputs")
    formulas: list[str] = []
    if not isinstance(outputs, list):
        return formulas
    for out in outputs:
        if isinstance(out, dict):
            formula = out.get("formula")
        else:
            formula = out
        if isinstance(formula, str) and formula.strip():
            formulas.append(formula.strip())
    return formulas


def validate_json_dataset(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return ["top-level JSON value must be a list"], {}

    ids: list[str] = []
    seen_ids: set[str] = set()
    seen_inputs: dict[str, str] = {}
    formula_count = 0
    multi_reading = 0
    parsed_ok = 0

    for index, item in enumerate(data, start=1):
        where = f"item #{index}"
        if not isinstance(item, dict):
            errors.append(f"{where}: expected object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not ID_RE.fullmatch(item_id):
            errors.append(f"{where}: invalid id {item_id!r}; expected exNN")
        else:
            ids.append(item_id)
            if item_id in seen_ids:
                errors.append(f"{where}: duplicate id {item_id}")
            seen_ids.add(item_id)
            number = int(ID_RE.fullmatch(item_id).group(1))  # type: ignore[union-attr]
            if number != index:
                warnings.append(f"{item_id}: id number does not match 1-based position {index}")
        inp = item.get("input")
        if not isinstance(inp, str) or not inp.strip():
            errors.append(f"{where}: missing or empty input")
        else:
            norm_input = normalize_text(inp)
            previous = seen_inputs.get(norm_input)
            if previous and previous != item_id:
                warnings.append(f"{item_id}: duplicate normalized input also used by {previous}")
            seen_inputs[norm_input] = str(item_id)
        formulas = item_outputs(item)
        if not formulas:
            errors.append(f"{item_id or where}: missing non-empty outputs[].formula")
            continue
        if len(formulas) > 1:
            multi_reading += 1
        local_seen: set[str] = set()
        for formula in formulas:
            formula_count += 1
            norm_formula = normalize_formula(formula)
            if norm_formula in local_seen:
                warnings.append(f"{item_id}: duplicate formula in outputs: {formula}")
            local_seen.add(norm_formula)
            try:
                FormulaParser(formula).parse()
                parsed_ok += 1
            except FormulaSyntaxError as exc:
                errors.append(f"{item_id}: formula syntax error: {formula!r}: {exc}")

    summary = {
        "items": len(data),
        "formulas": formula_count,
        "multi_reading_items": multi_reading,
        "parsed_formulas": parsed_ok,
        "sha256": sha256(path),
        "warnings": warnings,
    }
    return errors, summary


def extract_pdf_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext is required for --review-pdf validation") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed: {exc.stderr.strip()}") from exc


def validate_review_pdf(pdf_path: Path, dataset_path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    text = extract_pdf_text(pdf_path)
    pdf_ids = re.findall(r'"id"\s*:\s*"(ex\d+)"', text)
    pdf_id_set = set(pdf_ids)
    json_ids = [item.get("id") for item in data if isinstance(item, dict)]
    json_id_set = set(json_ids)

    missing_from_pdf = sorted(json_id_set - pdf_id_set)
    extra_in_pdf = sorted(pdf_id_set - json_id_set)
    if missing_from_pdf:
        errors.append(f"IDs present in JSON but missing from PDF: {missing_from_pdf[:20]}")
    if extra_in_pdf:
        errors.append(f"IDs present in PDF but missing from JSON: {extra_in_pdf[:20]}")

    # High-value regression check for the previously problematic ex334 item.
    ex334 = next((item for item in data if item.get("id") == "ex334"), None)
    if ex334:
        expected = normalize_formula(item_outputs(ex334)[0])
        loc = text.find('"id": "ex334"')
        window = normalize_formula(text[loc : loc + 1500]) if loc >= 0 else ""
        if expected not in window:
            errors.append("ex334 formula in JSON was not found in the review PDF text window")

    return errors, {
        "pdf_ids": len(pdf_ids),
        "unique_pdf_ids": len(pdf_id_set),
        "json_ids": len(json_ids),
        "unique_json_ids": len(json_id_set),
        "sha256": sha256(pdf_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="data/dataset_gold.json", help="Dataset JSON path")
    parser.add_argument("--review-pdf", help="Optional reviewer PDF exported from Drive")
    parser.add_argument("--write-report", help="Optional JSON report path")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    all_errors: list[str] = []
    report: dict[str, Any] = {"dataset": str(dataset_path)}

    try:
        dataset_errors, dataset_summary = validate_json_dataset(dataset_path)
        all_errors.extend(dataset_errors)
        report["dataset_summary"] = dataset_summary
    except Exception as exc:  # keep CLI useful for CI logs
        all_errors.append(f"failed to validate dataset JSON: {exc}")

    if args.review_pdf:
        try:
            pdf_errors, pdf_summary = validate_review_pdf(Path(args.review_pdf), dataset_path)
            all_errors.extend(pdf_errors)
            report["review_pdf_summary"] = pdf_summary
        except Exception as exc:
            all_errors.append(f"failed to validate review PDF: {exc}")

    report["ok"] = not all_errors
    report["errors"] = all_errors

    if args.write_report:
        Path(args.write_report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
