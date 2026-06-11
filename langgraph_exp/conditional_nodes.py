"""
Nodes for the conditional-debate LangGraph experiment.

This is deliberately close to debate_nodes.py, but adds a vision_gate_node and
agreement_check_node before deciding whether to run the debate loop.
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
ORCH_MODEL = os.getenv("LG_ORCH_MODEL", "qwen2.5vl:7b")

MAX_ROUNDS = int(os.getenv("LG_MAX_DEBATE_ROUNDS", "2"))
MAX_QUESTIONS = 3

VALID_LETTERS = set("ABCDE")


def _extract_letter(text: str) -> str:
    """Strict extraction from ANSWER: X format; fallback to UNKNOWN."""
    if not text:
        return "UNKNOWN"
    m = re.search(r"ANSWER:\s*([A-E])", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}) {v}" for k, v in options.items())


# def _extract_json(raw: str) -> dict[str, Any]:
#     """Best-effort JSON extraction for local models that may add extra text."""
#     if not raw:
#         return {}
#     start, end = raw.find("{"), raw.rfind("}")
#     block = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw
#     try:
#         return json.loads(block)
#     except json.JSONDecodeError:
#         return {}

def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction for local models that may add extra text.

    GUARANTEES a dict return. If parsing produces any non-dict type
    (string, list, number, null), returns an empty dict so downstream
    .get() calls are always safe.
    """
    if not raw:
        return {}
    start, end = raw.find("{"), raw.rfind("}")
    block = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw
    try:
        parsed = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return {}
    # Guard: json.loads can return str/list/int/None — only dict is usable here
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _safe_int(value: Any, default: int = 1, min_value: int = 1, max_value: int = 5) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, n))


def _parse_text_json(raw: str):
    obj = _extract_json(raw)
    ans = str(obj.get("answer", "")).strip().upper()[:1]
    ans = ans if ans in VALID_LETTERS else _extract_letter(raw)

    conf = _safe_int(obj.get("confidence", 1), default=1)
    reasoning = str(obj.get("reasoning", raw)).strip()

    qs = obj.get("visual_questions", []) or []
    if isinstance(qs, str):
        qs = [qs]
    qs = [str(q).strip() for q in qs if str(q).strip()][:MAX_QUESTIONS]

    return ans, conf, reasoning, qs


def _parse_vision_gate_json(raw: str, options: dict):
    obj = _extract_json(raw)

    ans = str(obj.get("answer", "")).strip().upper()[:1]
    ans = ans if ans in VALID_LETTERS else "UNKNOWN"

    conf = _safe_int(obj.get("confidence", 1), default=1)
    description = str(obj.get("image_description", "")).strip()
    reasoning = str(obj.get("reasoning", raw)).strip()
    support = obj.get("visual_support", {}) or {}
    if not isinstance(support, dict):
        support = {}

    # Ensure every option has a key, so W&B tables are easier to audit.
    normalized_support = {}
    for letter in options.keys():
        normalized_support[letter] = str(support.get(letter, "not_assessable"))

    return ans, conf, description, reasoning, normalized_support


# ─────────────────────────────────────────────────────────────────────────────
# Text initial: vignette + options, no image
# ─────────────────────────────────────────────────────────────────────────────

TEXT_INITIAL_SYSTEM = (
    "You are an experienced clinician answering a diagnostic multiple-choice "
    "question. You have the clinical vignette and options, but NOT the image. "
    "Give your current best answer and confidence. Also list specific visual "
    "questions you would ask a radiologist only if image findings could change "
    "your answer."
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
    ans, conf, reasoning, qs = _parse_text_json(raw)

    return {
        "text_answer": ans,
        "text_answer_initial": ans,
        "text_confidence": conf,
        "text_assessment": reasoning,
        "visual_questions": [qs],
        "visual_answers": [],
        "round": 0,
        "done_debating": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vision gate: image + options, no vignette
# ─────────────────────────────────────────────────────────────────────────────

VISION_GATE_SYSTEM = (
    "You are a medical imaging specialist. You will see a medical image and a "
    "set of answer options, but you will NOT see the clinical vignette. First "
    "describe the visible findings. Then judge which option, if any, is most "
    "supported by the image alone. If the image alone cannot distinguish the "
    "options, answer UNKNOWN with low confidence. Do not invent clinical history."
)


def vision_gate_node(state):
    user_text = f"""Options:
{_format_options(state['options'])}

Respond ONLY with JSON, no markdown:
{{
  "image_description": "<modality, anatomy, and salient visual findings>",
  "visual_support": {{
    "A": "supports | weak_support | refutes | not_assessable",
    "B": "supports | weak_support | refutes | not_assessable",
    "C": "supports | weak_support | refutes | not_assessable",
    "D": "supports | weak_support | refutes | not_assessable",
    "E": "supports | weak_support | refutes | not_assessable"
  }},
  "answer": "<single letter A-E or UNKNOWN>",
  "confidence": <integer 1-5>,
  "reasoning": "<1-2 sentences explaining image-only support>"
}}"""

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
    ans, conf, description, reasoning, support = _parse_vision_gate_json(raw, state["options"])

    return {
        "vision_answer": ans,
        "vision_confidence": conf,
        "image_description": description or reasoning,
        "vision_reasoning": reasoning,
        "visual_support": support,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agreement checker: deterministic routing decision
# ─────────────────────────────────────────────────────────────────────────────


def agreement_check_node(state):
    text_answer = state.get("text_answer", "UNKNOWN")
    vision_answer = state.get("vision_answer", "UNKNOWN")
    text_conf = int(state.get("text_confidence", 1))
    vision_conf = int(state.get("vision_confidence", 1))

    valid_text = text_answer in VALID_LETTERS
    valid_vision = vision_answer in VALID_LETTERS
    agreed = valid_text and valid_vision and text_answer == vision_answer

    # Rule 1: if both independently pick the same answer with decent confidence,
    # skip debate.
    if agreed and text_conf >= 3 and vision_conf >= 3:
        return {
            "agents_agreed": True,
            "debate_triggered": False,
            "routing_reason": "text and vision agreed with sufficient confidence",
            "final_mode": "direct_agreement",
        }

    # Rule 2: if image is not discriminative but text is highly confident,
    # skip debate and use text. This prevents ambiguous images from dragging down
    # a strong text-only diagnosis.
    if valid_text and (not valid_vision or vision_conf <= 2) and text_conf >= 4:
        return {
            "agents_agreed": False,
            "debate_triggered": False,
            "routing_reason": "vision uncertain/not assessable and text confidence high",
            "final_mode": "direct_text_high_conf",
        }

    # Otherwise, run directed debate.
    return {
        "agents_agreed": agreed,
        "debate_triggered": True,
        "routing_reason": "disagreement or insufficient confidence; running directed debate",
        "final_mode": "debated",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direct final: no extra model call
# ─────────────────────────────────────────────────────────────────────────────


def direct_final_node(state):
    mode = state.get("final_mode", "direct")

    if mode == "direct_agreement" and state.get("vision_answer") in VALID_LETTERS:
        ans = state["vision_answer"]
        reason = (
            f"Text and vision independently selected {ans}. "
            f"Text reasoning: {state.get('text_assessment', '')} "
            f"Image reasoning: {state.get('vision_reasoning', '')}"
        )
    else:
        ans = state.get("text_answer", "UNKNOWN")
        reason = (
            f"Using text answer because the text agent was highly confident and "
            f"the image gate was uncertain/not discriminative. Text reasoning: "
            f"{state.get('text_assessment', '')} Image reasoning: "
            f"{state.get('vision_reasoning', '')}"
        )

    return {
        "final_answer": ans,
        "final_output": f"ANSWER: {ans}\nREASONING: {reason}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Debate: vision answers text agent's questions
# ─────────────────────────────────────────────────────────────────────────────

VISION_QUERY_SYSTEM = (
    "You are a medical imaging specialist. A clinician will ask specific "
    "questions about the image. Answer factually based ONLY on what is visible. "
    "If a feature is not assessable, say so. Do NOT suggest a diagnosis and do "
    "NOT mention answer options."
)

DEFAULT_VISUAL_QUESTION = (
    "What are the most diagnostically significant findings visible in this image? "
    "Describe any abnormalities in detail, including location, appearance, and "
    "distinguishing characteristics."
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
# Debate: text revises after visual answer
# ─────────────────────────────────────────────────────────────────────────────

TEXT_REVISE_SYSTEM = (
    "You are the same clinician. The radiologist has answered your visual "
    "questions. Update your answer only if the visual findings warrant it. Ask "
    "further visual questions only if genuinely needed."
)


def text_revise_node(state):
    last_answers = state.get("visual_answers", [[]])[-1]
    va_text = "\n".join(last_answers) if last_answers else "[no visual answer returned]"

    user = f"""Clinical vignette:
{state['question']}

Options:
{_format_options(state['options'])}

Your previous answer: {state.get('text_answer', 'UNKNOWN')}
Your previous confidence: {state.get('text_confidence', 1)}
Your previous reasoning: {state.get('text_assessment', '')}

Radiologist's visual answer:
{va_text}

Respond ONLY with JSON, no markdown:
{{
  "answer": "<single letter A-E>",
  "confidence": <integer 1-5>,
  "reasoning": "<1-2 sentences; mention whether image changed your mind>",
  "visual_questions": ["<only if genuinely needed>"]
}}"""

    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": TEXT_REVISE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    ans, conf, reasoning, qs = _parse_text_json(raw)

    new_round = int(state.get("round", 0)) + 1
    done = (len(qs) == 0) or (new_round >= MAX_ROUNDS)

    return {
        "text_answer": ans,
        "text_confidence": conf,
        "text_assessment": reasoning,
        "visual_questions": state.get("visual_questions", []) + [qs],
        "round": new_round,
        "done_debating": done,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final orchestrator after debate only
# ─────────────────────────────────────────────────────────────────────────────

ORCH_SYSTEM = (
    "You are the senior physician making the final diagnostic decision. You have "
    "the clinical vignette, answer options, the image-only assessment, and the "
    "clinician's final assessment after visual consultation. Choose the best option."
)


def orchestrator_node(state):
    visual_answers_text = "\n".join(
        "\n".join(round_answers) for round_answers in state.get("visual_answers", [])
    )

    user = f"""Clinical vignette:
{state['question']}

Options:
{_format_options(state['options'])}

Image-only description:
{state.get('image_description', '[none]')}

Image-only option assessment:
Vision answer: {state.get('vision_answer', 'UNKNOWN')}
Vision confidence: {state.get('vision_confidence', 1)}
Vision reasoning: {state.get('vision_reasoning', '')}
Visual support by option: {state.get('visual_support', {})}

Directed visual consultation transcript:
{visual_answers_text or '[none]'}

Clinician's final assessment after consultation:
Answer: {state.get('text_answer', 'UNKNOWN')}
Confidence: {state.get('text_confidence', 1)}
Reasoning: {state.get('text_assessment', '')}

Select the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <2-3 sentences>"""

    resp = _client.chat.completions.create(
        model=ORCH_MODEL,
        messages=[
            {"role": "system", "content": ORCH_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content or ""
    return {"final_output": out, "final_answer": _extract_letter(out)}
