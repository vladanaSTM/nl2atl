"""Response parsing for LLM judge outputs."""

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class JudgeVerdict:
    decision: str
    reasoning: Optional[str] = None


def _clean_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r",\s*\}", "}", cleaned)
    cleaned = re.sub(r",\s*\]", "]", cleaned)
    return cleaned


def _json_object_candidates(text: str) -> List[str]:
    """Extract balanced object-like snippets from a model response."""
    candidates: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None

    return candidates


def _try_parse_mapping(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON/Python mapping, returning None for non-mapping values."""
    cleaned = _clean_json(text)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(cleaned)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_key_value_response(text: str) -> Optional[Dict[str, str]]:
    """Parse simple non-JSON judge responses such as 'correct: yes'."""
    decision_match = re.search(
        r"\bcorrect\s*[:=]\s*['\"]?(yes|no)['\"]?",
        text,
        flags=re.IGNORECASE,
    )
    if not decision_match:
        return None

    reasoning = ""
    reasoning_match = re.search(
        r"\breasoning\s*[:=]\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip().strip("\"'")

    return {"correct": decision_match.group(1), "reasoning": reasoning}


def parse_judge_response(response: str) -> JudgeVerdict:
    """Parse LLM judge response into structured verdict."""
    raw = str(response).strip()
    output_parts = re.split(r"\boutput\s*:", raw, flags=re.IGNORECASE)
    if len(output_parts) > 1:
        raw = output_parts[-1].strip()

    candidates = [raw]
    candidates.extend(reversed(_json_object_candidates(raw)))

    data = None
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        data = _try_parse_mapping(candidate)
        if data is not None:
            break

    if data is None:
        data = _parse_key_value_response(raw)

    if data is None:
        return JudgeVerdict(
            decision="no",
            reasoning="Judge response was not valid JSON.",
        )

    decision = str(data.get("correct", "no")).strip().lower()
    if decision not in ("yes", "no"):
        decision = "no"

    reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."
    return JudgeVerdict(decision=decision, reasoning=reasoning)


def extract_decision(response: str) -> str:
    """Extract decision from response text."""
    return parse_judge_response(response).decision
