"""
Nodes for the conservative conditional-debate LangGraph experiment.

The key difference from the earlier conditional debate:
  1. The vision gate is image-only: it does NOT see answer options or vignette.
  2. Text revision is guarded deterministically. A proposed answer change is
     accepted only when the model explicitly reports strong visual evidence
     that contradicts the previous answer and supports the new one.
  3. There is no final LLM orchestrator that can override the guarded answer.
     The final answer is the conservative text answer after the guard.
"""

import os
import re
import json
from typing import Any
from openai import OpenAI

_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=180.0,
)

VISION_MODEL = os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b")
TEXT_MODEL = os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b")

MAX_ROUNDS = int(os.getenv("LG_MAX_DEBATE_ROUNDS", "2"))
MAX_QUESTIONS = 3

VALID_LETTERS = set("ABCDE")
VALID_STRENGTHS = {"none", "weak", "moderate", "strong"}


def _extract_letter(text: str) -> str:
    """Strict extraction from ANSWER: X format; fallback to UNKNOWN."""
    if not text:
        return "UNKNOWN"
    m = re.search(r"ANSWER:\s*([A-E])", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}) {v}" for k, v in options.items())


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction for local models that may add extra text."""
    if not raw:
        return {}
    start, end = raw.find("{"), raw.rfind("}")
    block = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return {}


def _safe_int(value: Any, default: int = 1, min_value: int = 1, max_value: int = 5) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, n))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _force_valid_answer(ans: str, fallback: str = "A") -> str:
    """Final/text answers must be A-E. UNKNOWN is only for internal parse failures."""
    ans = (ans or "").strip().upper()[:1]
    return ans if ans in VALID_LETTERS else fallback


def _parse_text_initial_json(raw: str):
    obj = _extract_json(raw)
    ans = str(obj.get("answer", "")).strip().upper()[:1]
    ans = ans if ans in VALID_LETTERS else _extract_letter(raw)
    ans = _force_valid_answer(ans, fallback="A")

    conf = _safe_int(obj.get("confidence", 1), default=1)
    reasoning = str(obj.get("reasoning", raw)).strip()

    qs = obj.get("visual_questions", []) or []
    if isinstance(qs, str):
        qs = [qs]
    qs = [str(q).strip() for q in qs if str(q).strip()][:MAX_QUESTIONS]

    return ans, conf, reasoning, qs


def _parse_vision_gate_json(raw: str):
    obj = _extract_json(raw)
    description = str(obj.get("image_description", "")).strip()
    diagnosticity = _safe_int(obj.get("image_diagnosticity", 1), default=1)
    uncertainty = str(obj.get("uncertainty", "")).strip()
    reasoning = str(obj.get("reasoning", raw)).strip()
    return description, diagnosticity, uncertainty, reasoning


def _parse_text_revise_json(raw: str):
    obj = _extract_json(raw)
    ans = str(obj.get("answer", "")).strip().upper()[:1]
    ans = ans if ans in VALID_LETTERS else _extract_letter(raw)

    conf = _safe_int(obj.get("confidence", 1), default=1)
    reasoning = str(obj.get("reasoning", raw)).strip()

    strength = str(obj.get("visual_evidence_strength", "weak")).strip().lower()
    strength = strength if strength in VALID_STRENGTHS else "weak"

    contradicts = _safe_bool(obj.get("visual_contradicts_previous_answer", False))
    supports_new = _safe_bool(obj.get("visual_supports_new_answer", False))

    qs = obj.get("visual_questions", []) or []
    if isinstance(qs, str):
        qs = [qs]
    qs = [str(q).strip() for q in qs if str(q).strip()][:MAX_QUESTIONS]

    return ans, conf, reasoning, strength, contradicts, supports_new, qs


# ─────────────────────────────────────────────────────────────────────────────
# Text initial: vignette + options, no image
# ─────────────────────────────────────────────────────────────────────────────

TEXT_INITIAL_SYSTEM = (
    "You are an experienced clinician answering a diagnostic multiple-choice "
    "question. You have the clinical vignette and answer options, but NOT the "
    "image. Give your best text-only answer and confidence. Ask specific visual "
    "questions only if image findings could materially change the diagnosis."
)


def text_initial_node(state):
    user = f"""Clinical vignette:
{state['question']}

Options:
{_format_options(state['options'])}

Respond ONLY with JSON, no markdown:
{{
  "answer": "<single letter A-E>",
  "confidence": <integer 1-5>,
  "reasoning": "<1-2 sentences>",
  "visual_questions": ["<specific image question>", "..."]
}}

Confidence meaning:
1 = very uncertain, 3 = moderate, 5 = very confident.
Ask 0-3 visual questions."""

    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": TEXT_INITIAL_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    ans, conf, reasoning, qs = _parse_text_initial_json(raw)

    return {
        "text_answer_initial": ans,
        "text_initial_confidence": conf,
        "text_initial_assessment": reasoning,
        "text_answer": ans,
        "text_confidence": conf,
        "text_assessment": reasoning,
        "visual_questions": [qs],
        "visual_answers": [],
        "round": 0,
        "done_debating": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vision gate: image only, no vignette, no answer options
# ─────────────────────────────────────────────────────────────────────────────

VISION_GATE_SYSTEM = (
    "You are a medical imaging specialist. You will see a single medical image. "
    "You will NOT see the clinical vignette and you will NOT see answer options. "
    "Describe visible findings and estimate whether the image appears "
    "diagnostically useful. Do NOT guess a diagnosis."
)


def vision_gate_node(state):
    user_text = """Describe this medical image and estimate how diagnostically useful it is.

Respond ONLY with JSON, no markdown:
{
  "image_description": "<modality, anatomy/body region, and salient visible findings>",
  "image_diagnosticity": <integer 1-5>,
  "uncertainty": "<what cannot be determined from the image alone>",
  "reasoning": "<1-2 sentences explaining why the image is or is not diagnostically useful>"
}

image_diagnosticity meaning:
1 = image is nonspecific/not very useful
3 = image has useful but not decisive findings
5 = image has highly distinctive diagnostic findings"""

    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]

    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_GATE_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    raw = resp.choices[0].message.content or ""
    description, diagnosticity, uncertainty, reasoning = _parse_vision_gate_json(raw)

    return {
        "image_description": description or reasoning,
        "image_diagnosticity": diagnosticity,
        "image_uncertainty": uncertainty,
        "image_gate_reasoning": reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conservative router
# ─────────────────────────────────────────────────────────────────────────────


def conservative_route_node(state):
    text_conf = int(state.get("text_confidence", 1))
    diagnosticity = int(state.get("image_diagnosticity", 1))
    questions = state.get("visual_questions", [[]])[-1]

    # Rule 1: protect high-confidence text-only diagnoses. The earlier option-
    # ranking result showed weak vision top-answer accuracy, so high-confidence
    # text should not be exposed to debate unless you deliberately lower this.
    if text_conf >= 4:
        return {
            "debate_triggered": False,
            "final_mode": "direct_text_high_conf",
            "routing_reason": "text confidence >= 4; conservative policy skips debate",
        }

    # Rule 2: if the image itself looks low-value and text is at least moderate,
    # skip debate. Ambiguous images are a common source of harmful revisions.
    if text_conf >= 3 and diagnosticity <= 2:
        return {
            "debate_triggered": False,
            "final_mode": "direct_image_low_value",
            "routing_reason": "text confidence moderate and image diagnosticity <= 2",
        }

    # Rule 3: if text is uncertain, or the image seems useful, run one directed
    # consultation. If the text agent asked no questions, use a default broad one.
    return {
        "debate_triggered": True,
        "final_mode": "debated_guarded",
        "routing_reason": (
            f"text_confidence={text_conf}, image_diagnosticity={diagnosticity}, "
            f"n_initial_questions={len(questions)}; running guarded debate"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vision query: image + text-generated questions only
# ─────────────────────────────────────────────────────────────────────────────

VISION_QUERY_SYSTEM = (
    "You are a medical imaging specialist. A clinician will ask specific "
    "questions about the image. Answer factually based ONLY on what is visible. "
    "If a feature is not assessable, say so. Do NOT suggest a diagnosis and do "
    "NOT mention answer options."
)

DEFAULT_VISUAL_QUESTION = (
    "What are the most diagnostically significant findings visible in this image? "
    "Describe abnormalities in detail, including location, appearance, pattern, "
    "and distinguishing characteristics."
)


def vision_query_node(state):
    questions = state.get("visual_questions", [[]])[-1]
    if not questions:
        questions = [DEFAULT_VISUAL_QUESTION]

    q_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    content = [
        {"type": "text", "text": f"Answer these questions about the image:\n{q_block}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]

    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_QUERY_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    answer_text = resp.choices[0].message.content or ""

    return {
        "visual_questions": state.get("visual_questions", [])[:-1] + [questions],
        "visual_answers": state.get("visual_answers", []) + [[answer_text]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text revision proposal + deterministic conservative guard
# ─────────────────────────────────────────────────────────────────────────────

TEXT_REVISE_SYSTEM = (
    "You are the same clinician. The radiologist has answered your visual "
    "questions. You are CONSERVATIVE: keep your previous answer unless the visual "
    "evidence is strong, clearly contradicts the previous answer, and strongly "
    "supports a different option. Do not change answers based on weak or ambiguous "
    "image findings."
)


def _apply_conservative_guard(
    previous_answer: str,
    previous_confidence: int,
    proposed_answer: str,
    proposed_confidence: int,
    strength: str,
    contradicts_previous: bool,
    supports_new: bool,
):
    """Return (allowed, final_answer, guard_reason, blocked_change)."""
    previous_answer = _force_valid_answer(previous_answer, fallback="A")
    proposed_answer = proposed_answer if proposed_answer in VALID_LETTERS else previous_answer

    if proposed_answer == previous_answer:
        return True, previous_answer, "no answer change proposed", False

    strong_visual_change = (
        strength == "strong" and contradicts_previous and supports_new
    )

    if not strong_visual_change:
        return (
            False,
            previous_answer,
            "blocked: proposed change lacked strong contradictory visual evidence",
            True,
        )

    if previous_confidence >= 4 and proposed_confidence < 4:
        return (
            False,
            previous_answer,
            "blocked: previous text confidence high and proposed confidence < 4",
            True,
        )

    return True, proposed_answer, "allowed: strong contradictory visual evidence", False


def text_revise_node(state):
    last_answers = state.get("visual_answers", [[]])[-1]
    va_text = "\n".join(last_answers) if last_answers else "[no visual answer returned]"

    previous_answer = state.get("text_answer", state.get("text_answer_initial", "A"))
    previous_conf = int(state.get("text_confidence", state.get("text_initial_confidence", 1)))

    user = f"""Clinical vignette:
{state['question']}

Options:
{_format_options(state['options'])}

Your previous answer: {previous_answer}
Your previous confidence: {previous_conf}
Your previous reasoning: {state.get('text_assessment', '')}

Radiologist's visual answer:
{va_text}

Respond ONLY with JSON, no markdown:
{{
  "answer": "<single letter A-E; keep previous unless image evidence strongly justifies a change>",
  "confidence": <integer 1-5>,
  "reasoning": "<1-2 sentences; explicitly state whether the image changed your mind>",
  "visual_evidence_strength": "none | weak | moderate | strong",
  "visual_contradicts_previous_answer": <true or false>,
  "visual_supports_new_answer": <true or false>,
  "visual_questions": ["<ask only if another image detail is truly needed>"]
}}

Conservative rule:
Only change from {previous_answer} if the visual evidence is STRONG, contradicts {previous_answer}, and strongly supports another option.
If evidence is weak, moderate, nonspecific, or not assessable, keep {previous_answer}."""

    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": TEXT_REVISE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    proposed_ans, proposed_conf, proposed_reasoning, strength, contradicts, supports_new, qs = _parse_text_revise_json(raw)

    allowed, guarded_answer, guard_reason, blocked_change = _apply_conservative_guard(
        previous_answer=previous_answer,
        previous_confidence=previous_conf,
        proposed_answer=proposed_ans,
        proposed_confidence=proposed_conf,
        strength=strength,
        contradicts_previous=contradicts,
        supports_new=supports_new,
    )

    # If the guard blocks the change, keep previous confidence/reasoning but append
    # an audit note. If it allows, accept the proposed reasoning/confidence.
    if allowed and guarded_answer == proposed_ans:
        final_conf = proposed_conf
        final_reasoning = proposed_reasoning
    else:
        final_conf = previous_conf
        final_reasoning = (
            f"Kept previous answer {previous_answer}. Guard reason: {guard_reason}. "
            f"Proposed answer was {proposed_ans}; proposed reasoning: {proposed_reasoning}"
        )

    new_round = int(state.get("round", 0)) + 1

    # Stop if no further questions, max rounds reached, or a change was blocked.
    # The last condition prevents repeated attempts to persuade the text answer.
    done = (len(qs) == 0) or (new_round >= MAX_ROUNDS) or blocked_change

    return {
        "proposed_text_answer": proposed_ans if proposed_ans in VALID_LETTERS else "UNKNOWN",
        "proposed_text_confidence": proposed_conf,
        "proposed_text_assessment": proposed_reasoning,
        "visual_evidence_strength": strength,
        "visual_contradicts_previous": contradicts,
        "visual_supports_new_answer": supports_new,
        "revision_allowed": allowed,
        "revision_guard_reason": guard_reason,
        "guard_blocked_change": blocked_change,
        "text_answer": guarded_answer,
        "text_confidence": final_conf,
        "text_assessment": final_reasoning,
        "visual_questions": state.get("visual_questions", []) + [qs],
        "round": new_round,
        "done_debating": done,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final decision: deterministic, no extra LLM orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def final_decision_node(state):
    ans = _force_valid_answer(state.get("text_answer", state.get("text_answer_initial", "A")), fallback="A")

    if state.get("debate_triggered"):
        reason = (
            f"Conservative guarded debate used. Initial text answer: "
            f"{state.get('text_answer_initial')}; final guarded text answer: {ans}. "
            f"Revision allowed: {state.get('revision_allowed')}. "
            f"Guard reason: {state.get('revision_guard_reason', '')}. "
            f"Final reasoning: {state.get('text_assessment', '')}"
        )
    else:
        reason = (
            f"Debate skipped by conservative routing. Mode: {state.get('final_mode')}. "
            f"Routing reason: {state.get('routing_reason')}. "
            f"Text reasoning: {state.get('text_assessment', '')}. "
            f"Image diagnosticity: {state.get('image_diagnosticity')}"
        )

    return {
        "final_answer": ans,
        "final_output": f"ANSWER: {ans}\nREASONING: {reason}",
    }
