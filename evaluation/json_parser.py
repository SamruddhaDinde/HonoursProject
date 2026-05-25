"""
JSON parsing utility for Mode 3 structured communication.

Handles the common failure modes of LLM-generated JSON:
  - Markdown backtick fences (```json ... ```)
  - Preamble text before the JSON object
  - Trailing text after the JSON object
  - Missing or extra commas
  - Single retry on parse failure with a corrective prompt

Place this file at: /workspace/evaluation/json_parser.py
"""

import json
import re
from typing import Optional


#  Default schema (what we expect from specialists) ─

SPECIALIST_SCHEMA = {
    "answer": "?",
    "confidence": 0.5,
    "key_findings": [],
    "supporting_evidence": "",
    "alternative_considered": "?",
    "why_not_alternative": "",
}


def parse_specialist_json(raw_output: str) -> tuple[Optional[dict], str]:
    """Parse a specialist's structured JSON output.

    Tries progressively more aggressive extraction strategies.

    Returns:
        (parsed_dict, status) where status is one of:
          "clean"    — parsed on first try
          "stripped" — parsed after stripping markdown fences / preamble
          "partial"  — some fields extracted but JSON was malformed
          "failed"   — nothing usable extracted

    The returned dict always has all SPECIALIST_SCHEMA keys, with defaults
    for any missing fields. The "answer" field is normalised to uppercase.
    """
    if not raw_output or not raw_output.strip():
        return _default_result(), "failed"

    # Strategy 1: direct parse
    cleaned = _strip_fences(raw_output)
    result = _try_parse(cleaned)
    if result is not None:
        return _normalise(result), "clean"

    # Strategy 2: extract the first { ... } block
    extracted = _extract_json_block(raw_output)
    if extracted:
        result = _try_parse(extracted)
        if result is not None:
            return _normalise(result), "stripped"

    # Strategy 3: try fixing common JSON errors
    if extracted:
        fixed = _fix_common_errors(extracted)
        result = _try_parse(fixed)
        if result is not None:
            return _normalise(result), "stripped"

    # Strategy 4: regex extraction of individual fields (last resort)
    partial = _regex_extract(raw_output)
    if partial.get("answer", "?") != "?":
        return _normalise(partial), "partial"

    return _default_result(), "failed"


def build_retry_prompt(original_prompt: str, bad_output: str) -> str:
    """Build a corrective prompt for a single retry after JSON parse failure.

    Shorter and more directive than the original prompt — the model has
    already seen the case, it just needs to fix the format.
    """
    return f"""Your previous response was not valid JSON. Here is what you produced:

{bad_output[:500]}

Please respond with ONLY a valid JSON object matching this exact schema.
Do not include any text outside the JSON object. No markdown backticks.

{{
  "answer": "<single letter A-E>",
  "confidence": <number between 0.0 and 1.0>,
  "key_findings": ["<finding 1>", "<finding 2>"],
  "supporting_evidence": "<one sentence>",
  "alternative_considered": "<single letter A-E>",
  "why_not_alternative": "<one sentence>"
}}"""


#  Internal helpers ─

def _strip_fences(text: str) -> str:
    """Remove markdown code fences and common preamble patterns."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Remove common preambles like "Here is the JSON:" or "Sure, here's..."
    text = re.sub(
        r"^(?:here\s+is|sure|okay|certainly)[^{]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Find the first { ... } block using brace matching."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unmatched braces — try adding closing brace
    return text[start:] + "}"


def _fix_common_errors(text: str) -> str:
    """Fix common JSON syntax errors from LLMs."""
    # Trailing comma before closing brace/bracket
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Single quotes to double quotes (risky but often necessary)
    # Only do this if there are no double quotes at all
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _try_parse(text: str) -> Optional[dict]:
    """Attempt JSON parse, return None on failure."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _regex_extract(text: str) -> dict:
    """Last-resort field extraction using regex patterns."""
    result = dict(SPECIALIST_SCHEMA)

    # Answer: look for "answer": "X" or ANSWER: X
    ans_match = re.search(
        r'"answer"\s*:\s*"([A-E])"', text, re.IGNORECASE
    ) or re.search(
        r"ANSWER:\s*([A-E])", text, re.IGNORECASE
    ) or re.search(
        r"\b([A-E])\b", text[:200]
    )
    if ans_match:
        result["answer"] = ans_match.group(1).upper()

    # Confidence: look for "confidence": 0.XX
    conf_match = re.search(
        r'"confidence"\s*:\s*(0?\.\d+|1\.0|0|1)', text
    )
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    return result


def _normalise(result: dict) -> dict:
    """Ensure all expected keys exist and answer is uppercase."""
    normalised = dict(SPECIALIST_SCHEMA)
    normalised.update(result)

    # Normalise answer
    ans = str(normalised.get("answer", "?")).strip().upper()
    if len(ans) == 1 and ans in "ABCDE":
        normalised["answer"] = ans
    else:
        normalised["answer"] = "?"

    # Normalise confidence
    try:
        conf = float(normalised.get("confidence", 0.5))
        normalised["confidence"] = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        normalised["confidence"] = 0.5

    # Ensure key_findings is a list
    kf = normalised.get("key_findings", [])
    if isinstance(kf, str):
        normalised["key_findings"] = [kf]
    elif not isinstance(kf, list):
        normalised["key_findings"] = []

    # Ensure alternative_considered is a string, not a list
    alt = normalised.get("alternative_considered", "?")
    if isinstance(alt, list):
        normalised["alternative_considered"] = alt[0] if alt else "?"
    else:
        normalised["alternative_considered"] = str(alt).strip().upper()[:1] or "?"

    # Ensure why_not_alternative is a string
    why = normalised.get("why_not_alternative", "")
    if isinstance(why, list):
        normalised["why_not_alternative"] = "; ".join(str(w) for w in why)
    elif not isinstance(why, str):
        normalised["why_not_alternative"] = str(why)
        
    return normalised


def _default_result() -> dict:
    """Return a default dict when all parsing strategies fail."""
    return dict(SPECIALIST_SCHEMA)