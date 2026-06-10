"""
Nodes for the directed-debate orchestrator.

Reuses the model client and helpers pattern from nodes.py. The text agent
emits a JSON block with its answer AND a list of visual questions; the vision
node answers each question from the image alone; the text agent revises.
"""

import os
import re
import json
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
MAX_QUESTIONS = 3   # cap visual questions per round to bound cost


def _extract_letter(text: str) -> str:
    if not text:
        return "UNKNOWN"
    m = re.search(r"ANSWER:\s*([A-E])", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}) {v}" for k, v in options.items())


def _parse_text_json(raw: str):
    """Extract {answer, reasoning, visual_questions[]} from the text agent.
    Tolerant: falls back to letter extraction and empty questions on failure."""
    block = raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = raw[start:end + 1]
    try:
        obj = json.loads(block)
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        ans = ans if ans in "ABCDE" else _extract_letter(raw)
        qs = obj.get("visual_questions", []) or []
        if isinstance(qs, str):
            qs = [qs]
        qs = [str(q).strip() for q in qs if str(q).strip()][:MAX_QUESTIONS]
        reasoning = str(obj.get("reasoning", "")).strip()
        return ans, reasoning, qs
    except (json.JSONDecodeError, ValueError, AttributeError):
        return _extract_letter(raw), raw.strip(), []


# ── Baseline full description (round 0), so orchestrator always has one ──

VISION_DESCRIBE_SYSTEM = (
    "You are a medical imaging specialist. Describe ONLY what is visible in the "
    "image: modality, anatomy, and all salient findings. Do NOT guess a diagnosis. "
    "Do NOT mention multiple-choice options."
)


def vision_describe_node(state):
    content = [
        {"type": "text", "text": "Describe this medical image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]
    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": VISION_DESCRIBE_SYSTEM},
                  {"role": "user", "content": content}],
    )
    return {"image_description": resp.choices[0].message.content or ""}


# ── Text agent: initial answer + visual questions ──

TEXT_INITIAL_SYSTEM = (
    "You are an experienced clinician answering a diagnostic multiple-choice "
    "question. You have the vignette and options but NOT the image. A radiologist "
    "can answer SPECIFIC questions about the image for you. Give your current best "
    "answer, then list specific visual questions that can help you with the diagnosis."
)

TEXT_INITIAL_TEMPLATE = """Clinical vignette:
{question}

Options:
{options}

Respond ONLY with JSON, no other text:
{{
  "answer": "<single letter A-E, your current best guess>",
  "reasoning": "<1-2 sentences>",
  "visual_questions": ["<specific question about the image>", "..."]
}}
Ask 0-3 questions."""


def text_initial_node(state):
    user = TEXT_INITIAL_TEMPLATE.format(
        question=state["question"], options=_format_options(state["options"]))
    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "system", "content": TEXT_INITIAL_SYSTEM},
                  {"role": "user", "content": user}],
    )
    ans, reasoning, qs = _parse_text_json(resp.choices[0].message.content or "")
    # Enforce a minimum consultation: if the text agent asked nothing, inject a
    # default probe so vision still contributes a targeted finding in round 1.
    # (The graph also forces round-0 -> vision_query; this ensures the query is
    # not empty.) This keeps every case comparable: all get >=1 real consult.
    if not qs:
        qs = ["What are the most diagnostically significant findings visible "
              "in this image? Describe any abnormalities in detail, including "
              "their location, appearance, and distinguishing characteristics."]
    return {
        "text_answer": ans,
        "text_answer_initial": ans,
        "text_assessment": reasoning,
        "visual_questions": [qs],     # round 0 questions
        "visual_answers": [],         # loop-control init (owned here, not in vision_describe)
        "round": 0,                   # so the parallel branches write disjoint keys
        "done_debating": False,       # never terminate before the first consult
    }


# ── Vision query: answer the text agent's specific questions ──

VISION_QUERY_SYSTEM = (
    "You are a medical imaging specialist. A clinician will ask you SPECIFIC "
    "questions about this image. Answer each one factually based ONLY on what you "
    "can see. If a feature is not assessable, say so. Do NOT suggest a diagnosis "
    "and do NOT mention any answer options."
)


def vision_query_node(state):
    current_round = state["round"]
    questions = state["visual_questions"][-1] if state["visual_questions"] else []
    if not questions:
        return {"visual_answers": state.get("visual_answers", []) + [[]]}

    q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    content = [
        {"type": "text", "text": f"Answer these questions about the image:\n{q_block}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]
    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": VISION_QUERY_SYSTEM},
                  {"role": "user", "content": content}],
    )
    answer_text = resp.choices[0].message.content or ""
    return {"visual_answers": state.get("visual_answers", []) + [[answer_text]]}


# ── Text revise: update answer given the visual answers, maybe ask more ──

TEXT_REVISE_SYSTEM = (
    "You are the same clinician. The radiologist has answered your visual "
    "questions. Update your answer if the visual findings warrant it. You may ask "
    "further questions ONLY if genuinely needed; otherwise return an empty list to "
    "conclude."
)

TEXT_REVISE_TEMPLATE = """Clinical vignette:
{question}

Options:
{options}

Your previous answer: {prev_answer}
Your previous reasoning: {prev_reasoning}

Radiologist's answers to your questions:
{visual_answers}

Respond ONLY with JSON:
{{
  "answer": "<single letter A-E, possibly revised>",
  "reasoning": "<1-2 sentences, note if the image changed your mind>",
  "visual_questions": ["<only if truly needed>"]
}}"""


def text_revise_node(state):
    last_answers = state["visual_answers"][-1] if state["visual_answers"] else []
    va_text = "\n".join(last_answers) if last_answers else "[no answers returned]"
    user = TEXT_REVISE_TEMPLATE.format(
        question=state["question"], options=_format_options(state["options"]),
        prev_answer=state["text_answer"], prev_reasoning=state["text_assessment"],
        visual_answers=va_text)
    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "system", "content": TEXT_REVISE_SYSTEM},
                  {"role": "user", "content": user}],
    )
    ans, reasoning, qs = _parse_text_json(resp.choices[0].message.content or "")
    new_round = state["round"] + 1
    done = (len(qs) == 0) or (new_round >= MAX_ROUNDS)
    return {
        "text_answer": ans,
        "text_assessment": reasoning,
        "visual_questions": state["visual_questions"] + [qs],
        "round": new_round,
        "done_debating": done,
    }


# ── Orchestrator: final fusion ──

ORCH_SYSTEM = (
    "You are the senior physician making the final decision. You receive the "
    "clinician's final assessment (after consulting a radiologist about the image) "
    "and the radiologist's neutral description. Choose the best option."
)

ORCH_TEMPLATE = """Clinical vignette:
{question}

Options:
{options}

Radiologist's image description:
{description}

Clinician's final assessment (after visual consultation): answer {answer}
Reasoning: {reasoning}

Select the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <2-3 sentences>"""


def orchestrator_node(state):
    user = ORCH_TEMPLATE.format(
        question=state["question"], options=_format_options(state["options"]),
        description=state.get("image_description", "[none]"),
        answer=state.get("text_answer", "UNKNOWN"),
        reasoning=state.get("text_assessment", ""))
    resp = _client.chat.completions.create(
        model=ORCH_MODEL,
        messages=[{"role": "system", "content": ORCH_SYSTEM},
                  {"role": "user", "content": user}],
    )
    out = resp.choices[0].message.content or ""
    return {"final_output": out, "final_answer": _extract_letter(out)}