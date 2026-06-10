"""
Nodes for the orchestrator workflow.

Topology:
    vision_describe_node ─┐
                          ├─> orchestrator_node ─> END
    text_search_node ─────┘

Design notes / guardrails:
  * Vision node DESCRIBES ONLY. It never sees the question or options, so it
    cannot "answer". This is the point of the experiment: does a pure visual
    description, fused by the orchestrator, beat a vision agent that guesses?
  * Text node sees vignette + options, may search the web (if enabled), and
    gives a tentative answer. Every query/snippet is recorded in state.
  * Orchestrator sees the description + the text assessment + the options and
    makes the final call. It does NOT see the raw image (it reasons over the
    description) — keeping the modality boundary clean for analysis.
  * ground_truth is in state for logging only and is NEVER put in a prompt.

Models are called through Ollama's OpenAI-compatible endpoint, matching your
existing setup (OPENAI_BASE_URL). Vision uses a VLM; text/orchestrator use a
text model. Set the model names in run_orchestrator.py.
"""

import os
import re
from openai import OpenAI

from .tools import run_search

_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=180.0,
)

# Model names are injected via env so a run can swap MedGemma <-> Qwen without
# touching code (supports your planned Qwen vision swap as a one-line change).
VISION_MODEL = os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b")
TEXT_MODEL = os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b")
ORCH_MODEL = os.getenv("LG_ORCH_MODEL", "qwen2.5vl:7b")


def _extract_letter(text: str) -> str:
    if not text:
        return "UNKNOWN"
    m = re.search(r"ANSWER:\s*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"  # NB: no bare-letter fallback — see analysis on extraction noise


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}) {v}" for k, v in options.items())


# ─────────────────────────────────────────────────────────────────────────
# Vision node — describe only, no question, no options
# ─────────────────────────────────────────────────────────────────────────

VISION_SYSTEM = (
    "You are a medical imaging specialist. You will be shown a single medical "
    "image. It may be a radiograph, CT, MRI, ultrasound, histopathology slide, "
    "fundus photo, ECG, or a clinical photograph of a body region or lesion. "
    "Describe ONLY what is visible: imaging modality, anatomical region, and all "
    "salient visual findings (location, size, shape, colour, texture, density, "
    "distribution, any abnormalities). Be specific and systematic. Do NOT guess a "
    "diagnosis. Describe only."
)


def vision_describe_node(state):
    content = [
        {"type": "text", "text": "Describe this medical image in detail."},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
    ]
    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    return {"image_description": resp.choices[0].message.content or ""}


# ─────────────────────────────────────────────────────────────────────────
# Text node — vignette + options, optional web search
# ─────────────────────────────────────────────────────────────────────────

TEXT_SYSTEM = (
    "You are an experienced clinician working through a diagnostic multiple-choice "
    "question. You are given the clinical vignette and the answer options onw of which is the right diagnosis, but NO "
    "image. Reason from the clinical history toward the most likely option."
)


def text_search_node(state):
    options_str = _format_options(state["options"])
    queries, snippets = [], []
    search_block = ""

    if state.get("use_web_search"):
        # Bias the query toward disease/finding knowledge rather than the verbatim
        # case (which is more likely to hit NEJM quiz pages and leak the answer).
        # We search ONE query built from the option set, to bound cost and leakage.
        query = "clinical features differentiating: " + "; ".join(state["options"].values())
        formatted, raw = run_search(query, max_results=3)
        queries.append(query)
        snippets.extend(raw)
        search_block = (
            "\n\nReference information from a web search (may be noisy; use "
            "critically, do not assume it names the answer):\n" + formatted
        )

    user = f"""Clinical vignette:
{state['question']}

Options:
{options_str}{search_block}

Decide the single most likely option.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <2-3 sentences of clinical reasoning>"""

    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": TEXT_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content or ""
    return {
        "text_assessment": out,
        "text_answer": _extract_letter(out),
        "search_queries": queries,
        "search_snippets": snippets,
    }


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator node — fuse description + text assessment, decide
# ─────────────────────────────────────────────────────────────────────────

ORCH_SYSTEM = (
    "You are the senior physician making the final diagnostic decision. You "
    "receive (1) a radiology/imaging description from an imaging specialist who "
    "did NOT know the question, and (2) a clinical assessment from a colleague "
    "who reasoned from the patient history but did NOT see the image. Integrate "
    "the visual findings with the clinical reasoning and choose the best option."
)


def orchestrator_node(state):
    options_str = _format_options(state["options"])
    user = f"""Clinical vignette:
{state['question']}

Options:
{options_str}

Imaging specialist's description of the image (no diagnosis given):
{state.get('image_description', '[none]')}

Clinical colleague's assessment:
{state.get('text_assessment', '[none]')}

Integrate the imaging findings with the clinical reasoning and select the most
likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <2-3 sentences integrating image and clinical findings>"""

    resp = _client.chat.completions.create(
        model=ORCH_MODEL,
        messages=[
            {"role": "system", "content": ORCH_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content or ""
    return {"final_output": out, "final_answer": _extract_letter(out)}