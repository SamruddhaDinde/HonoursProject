"""
Nodes for the option-ranking fusion experiment.

Topology:
    START ─┬─> text_rank_node ───┐
           └─> vision_rank_node ──┤
                                  └─> fusion_node ─> END

Design:
  * The text agent ranks all options using only clinical vignette + options.
  * The vision agent ranks/labels all options using only image + options.
  * The fusion agent combines the two structured rankings and chooses A-E.

This is not a debate graph. There is no back-and-forth communication between
text and vision, so it tests whether structured option-level evidence is better
than free-form fusion or debate.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple

from openai import OpenAI

_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=180.0,
)

VISION_MODEL = os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b")
TEXT_MODEL = os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b")
ORCH_MODEL = os.getenv("LG_ORCH_MODEL", "medgemma1.5:4b")

VALID_LETTERS = ("A", "B", "C", "D", "E")
SUPPORT_LABELS = {"supports", "weak_support", "neutral", "refutes", "not_assessable"}


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}) {v}" for k, v in options.items())


def _extract_letter(text: str) -> str:
    """Extract a single A-E answer from common model output patterns."""
    if not text:
        return "UNKNOWN"

    patterns = [
        r"ANSWER\s*:\s*([A-E])",
        r"FINAL_ANSWER\s*:\s*([A-E])",
        r"top_answer\s*[\"']?\s*[:=]\s*[\"']?([A-E])",
        r"top_visual_option\s*[\"']?\s*[:=]\s*[\"']?([A-E])",
        r"answer\s*[\"']?\s*[:=]\s*[\"']?([A-E])",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def _json_block(raw: str) -> dict:
    """Extract the outermost JSON object from a model response."""
    if not raw:
        raise ValueError("empty model output")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(raw[start:end + 1])


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _normalise_text_scores(raw_scores: Any, options: dict, top_answer: str) -> Dict[str, int]:
    """Return {A-E: 1-5}. Missing/malformed scores get conservative defaults."""
    scores: Dict[str, int] = {}
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    for letter in options.keys():
        if letter in VALID_LETTERS:
            scores[letter] = _clamp_int(raw_scores.get(letter), 1, 5, 1)
    if top_answer in scores and max(scores.values(), default=0) < 5:
        scores[top_answer] = 5
    return scores


def _normalise_vision_scores(raw_scores: Any, options: dict, top_answer: str) -> Dict[str, int]:
    """Return {A-E: 0-5}. 0 means not visually assessable."""
    scores: Dict[str, int] = {}
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    for letter in options.keys():
        if letter in VALID_LETTERS:
            scores[letter] = _clamp_int(raw_scores.get(letter), 0, 5, 0)
    if top_answer in scores and max(scores.values(), default=0) < 4:
        scores[top_answer] = 4
    return scores


def _normalise_support(raw_support: Any, options: dict) -> Dict[str, str]:
    support: Dict[str, str] = {}
    raw_support = raw_support if isinstance(raw_support, dict) else {}
    for letter in options.keys():
        if letter not in VALID_LETTERS:
            continue
        label = str(raw_support.get(letter, "not_assessable")).strip().lower().replace(" ", "_")
        if label not in SUPPORT_LABELS:
            label = "not_assessable"
        support[letter] = label
    return support


def _best_score_letter(scores: Dict[str, int], fallback: str = "A") -> str:
    if not scores:
        return fallback
    valid_scores = {k: v for k, v in scores.items() if k in VALID_LETTERS}
    if not valid_scores:
        return fallback
    return max(valid_scores.items(), key=lambda kv: kv[1])[0]


def _assert_no_image_payload(messages: list[dict]) -> None:
    text = str(messages)
    assert "data:image" not in text, "Text/fusion prompt accidentally contains image payload"
    assert "image_b64" not in text, "Text/fusion prompt accidentally contains image_b64 key"


def _assert_no_vignette_or_options(messages: list[dict], state: dict, *, allow_options: bool) -> None:
    text = str(messages)
    assert state.get("question", "__missing_question__") not in text, (
        "Vision prompt accidentally contains clinical vignette/question"
    )
    if not allow_options:
        for option in state.get("options", {}).values():
            assert str(option) not in text, "Vision prompt accidentally contains answer options"


# ──────────────────────────────────────────────────────────────────────────────
# Text rank node: vignette + options, NO image
# ──────────────────────────────────────────────────────────────────────────────

TEXT_RANK_SYSTEM = (
    "You are an experienced clinician answering a diagnostic multiple-choice "
    "question. You are given the clinical vignette and the five answer options, "
    "but you do NOT see the image. Rank every option using clinical evidence only."
)

TEXT_RANK_TEMPLATE = """Clinical vignette:
{question}

Options:
{options}

Return ONLY valid JSON with this schema:
{{
  "top_answer": "<single letter A-E>",
  "confidence": <integer 1-5>,
  "option_scores": {{"A": <1-5>, "B": <1-5>, "C": <1-5>, "D": <1-5>, "E": <1-5>}},
  "reasoning": "<2-4 sentences explaining the clinical differential>"
}}

Scoring guide:
1 = very unlikely clinically, 3 = plausible, 5 = most likely clinically.
You must score all five options and choose one top_answer."""


def text_rank_node(state: dict) -> dict:
    user = TEXT_RANK_TEMPLATE.format(
        question=state["question"],
        options=_format_options(state["options"]),
    )
    messages = [
        {"role": "system", "content": TEXT_RANK_SYSTEM},
        {"role": "user", "content": user},
    ]
    _assert_no_image_payload(messages)

    resp = _client.chat.completions.create(model=TEXT_MODEL, messages=messages)
    raw = resp.choices[0].message.content or ""

    parse_failed = False
    try:
        obj = _json_block(raw)
        top = str(obj.get("top_answer", "")).strip().upper()[:1]
        if top not in VALID_LETTERS:
            top = _extract_letter(raw)
        if top not in VALID_LETTERS:
            top = "A"  # deterministic forced fallback; logged by parse_failed
            parse_failed = True
        scores = _normalise_text_scores(obj.get("option_scores", {}), state["options"], top)
        confidence = _clamp_int(obj.get("confidence"), 1, 5, scores.get(top, 3))
        reasoning = str(obj.get("reasoning", "")).strip()
    except Exception:  # noqa: BLE001 - deliberately tolerant for long batch runs
        parse_failed = True
        top = _extract_letter(raw)
        if top not in VALID_LETTERS:
            top = "A"
        scores = {letter: (5 if letter == top else 1) for letter in state["options"].keys() if letter in VALID_LETTERS}
        confidence = 3
        reasoning = raw.strip()

    return {
        "text_raw_output": raw,
        "text_top_answer": top,
        "text_confidence": confidence,
        "text_scores": scores,
        "text_reasoning": reasoning,
        "text_parse_failed": parse_failed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Vision rank node: image + options, NO vignette
# ──────────────────────────────────────────────────────────────────────────────

VISION_RANK_SYSTEM = (
    "You are a medical imaging specialist. You will see one medical image and "
    "five candidate diagnostic labels. You do NOT see the clinical vignette. "
    "Assess only whether the visible image findings support, weakly support, "
    "are neutral for, refute, or are not assessable for each option. Do not use "
    "outside clinical context that is not visible in the image."
)

VISION_RANK_TEXT = """Candidate answer options:
{options}

Look at the image and return ONLY valid JSON with this schema:
{{
  "top_visual_option": "<A-E or UNKNOWN>",
  "confidence": <integer 0-5>,
  "option_scores": {{"A": <0-5>, "B": <0-5>, "C": <0-5>, "D": <0-5>, "E": <0-5>}},
  "option_support": {{"A": "supports|weak_support|neutral|refutes|not_assessable", "B": "...", "C": "...", "D": "...", "E": "..."}},
  "key_visual_findings": "<specific visible findings only>",
  "reasoning": "<1-3 sentences explaining visual support only>"
}}

Scoring guide:
0 = not visually assessable, 1 = visually refuted/unlikely, 3 = visually compatible but nonspecific, 5 = strongly visually supported.
Use UNKNOWN for top_visual_option if no option is clearly supported by the image alone."""


def vision_rank_node(state: dict) -> dict:
    text = VISION_RANK_TEXT.format(options=_format_options(state["options"]))
    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]
    messages = [
        {"role": "system", "content": VISION_RANK_SYSTEM},
        {"role": "user", "content": content},
    ]
    # This is intentionally option-aware, but must not include the vignette.
    _assert_no_vignette_or_options(messages, state, allow_options=True)

    resp = _client.chat.completions.create(model=VISION_MODEL, messages=messages)
    raw = resp.choices[0].message.content or ""

    parse_failed = False
    try:
        obj = _json_block(raw)
        top = str(obj.get("top_visual_option", "UNKNOWN")).strip().upper()[:1]
        if top not in VALID_LETTERS:
            top_raw = str(obj.get("top_visual_option", "UNKNOWN")).strip().upper()
            top = "UNKNOWN" if "UNKNOWN" in top_raw else _extract_letter(raw)
        if top not in VALID_LETTERS:
            top = "UNKNOWN"
        scores = _normalise_vision_scores(obj.get("option_scores", {}), state["options"], top)
        # If all scores are weak/nonspecific, mark top as UNKNOWN even if model forced one.
        if top in VALID_LETTERS and scores.get(top, 0) < 4:
            top = "UNKNOWN"
        confidence = _clamp_int(obj.get("confidence"), 0, 5, scores.get(top, 0) if top in VALID_LETTERS else 0)
        support = _normalise_support(obj.get("option_support", {}), state["options"])
        key_findings = str(obj.get("key_visual_findings", "")).strip()
        reasoning = str(obj.get("reasoning", "")).strip()
    except Exception:  # noqa: BLE001
        parse_failed = True
        top = _extract_letter(raw)
        if top not in VALID_LETTERS:
            top = "UNKNOWN"
        scores = {letter: (4 if letter == top else 0) for letter in state["options"].keys() if letter in VALID_LETTERS}
        confidence = 2 if top in VALID_LETTERS else 0
        support = {letter: ("weak_support" if letter == top else "not_assessable") for letter in state["options"].keys() if letter in VALID_LETTERS}
        key_findings = raw.strip()
        reasoning = raw.strip()

    return {
        "vision_raw_output": raw,
        "vision_top_answer": top,
        "vision_confidence": confidence,
        "vision_scores": scores,
        "vision_support": support,
        "key_visual_findings": key_findings,
        "vision_reasoning": reasoning,
        "vision_parse_failed": parse_failed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fusion node: structured text ranking + structured vision ranking, NO raw image
# ──────────────────────────────────────────────────────────────────────────────

FUSION_SYSTEM = (
    "You are the senior physician making the final diagnostic decision. You see "
    "the clinical vignette, the answer options, a clinician's text-only ranking, "
    "and an imaging specialist's option-level visual assessment. You do NOT see "
    "the raw image. Treat the text ranking as the clinical prior and the visual "
    "ranking as evidence only when the visual findings are specific and assessable."
)

FUSION_TEMPLATE = """Clinical vignette:
{question}

Options:
{options}

Text-only clinician ranking:
Top answer: {text_top}
Confidence: {text_confidence}/5
Option scores: {text_scores}
Reasoning: {text_reasoning}

Image-only option-level visual assessment:
Top visual option: {vision_top}
Confidence: {vision_confidence}/5
Visual option scores: {vision_scores}
Visual support labels: {vision_support}
Key visual findings: {key_visual_findings}
Visual reasoning: {vision_reasoning}

Decision rules:
- Choose exactly one of A, B, C, D, or E.
- Do not choose an option only because vision marked it compatible; compatibility is weaker than strong visual support.
- If visual evidence is not assessable or nonspecific, rely mainly on the clinical text ranking.
- If visual evidence strongly supports a different option and matches the vignette, you may override the text ranking.

Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <2-4 sentences explaining how text ranking and visual evidence were combined>"""


def fusion_node(state: dict) -> dict:
    user = FUSION_TEMPLATE.format(
        question=state["question"],
        options=_format_options(state["options"]),
        text_top=state.get("text_top_answer", "UNKNOWN"),
        text_confidence=state.get("text_confidence", 0),
        text_scores=json.dumps(state.get("text_scores", {}), sort_keys=True),
        text_reasoning=state.get("text_reasoning", ""),
        vision_top=state.get("vision_top_answer", "UNKNOWN"),
        vision_confidence=state.get("vision_confidence", 0),
        vision_scores=json.dumps(state.get("vision_scores", {}), sort_keys=True),
        vision_support=json.dumps(state.get("vision_support", {}), sort_keys=True),
        key_visual_findings=state.get("key_visual_findings", ""),
        vision_reasoning=state.get("vision_reasoning", ""),
    )
    messages = [
        {"role": "system", "content": FUSION_SYSTEM},
        {"role": "user", "content": user},
    ]
    _assert_no_image_payload(messages)

    resp = _client.chat.completions.create(model=ORCH_MODEL, messages=messages)
    raw = resp.choices[0].message.content or ""

    final = _extract_letter(raw)
    if final not in VALID_LETTERS:
        # Forced deterministic fallback for evaluation, logged in final_output.
        text_top = state.get("text_top_answer", "UNKNOWN")
        if text_top in VALID_LETTERS:
            final = text_top
        else:
            final = _best_score_letter(state.get("text_scores", {}), fallback="A")
        raw = raw + f"\n\n[FORMAT_FALLBACK_USED: final_answer={final}]"

    return {
        "final_output": raw,
        "final_answer": final,
        "fusion_changed_from_text": final != state.get("text_top_answer"),
    }
