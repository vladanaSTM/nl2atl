"""Response parsing for LLM judge outputs."""

import ast
import json
import re
from dataclasses import dataclass
from typing import Optional


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


def parse_judge_response(response: str) -> JudgeVerdict:
    """Parse LLM judge response into structured verdict."""
    raw = response
    if "output:" in raw:
        raw = raw.split("output:")[-1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
        data = None

        for candidate in reversed(candidates):
            cleaned = _clean_json(candidate)
            try:
                data = json.loads(cleaned)
                break
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(cleaned)
                    if isinstance(data, dict):
                        break
                    data = None
                except Exception:
                    continue

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
