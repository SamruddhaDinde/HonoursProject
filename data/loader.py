"""
NEJM Image Challenge dataset loader.

Loads cases from the local cx0/nejm-image-challenge JSON, filters to those
with substantive clinical context, and pairs them with locally cached images.

Expected layout (paths shown inside the Singularity container):
    /workspace/dataset/
    ├── image_challenge_dataset_20231223.json
    └── images/
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...

Override the dataset directory by setting NEJM_DATASET_DIR in .env.
"""
from evaluation.json_parser import parse_specialist_json
from evaluation.evaluator import extract_answer

import os
import json
import base64
import random
from io import BytesIO
from pathlib import Path

from PIL import Image


# Defaults to the Singularity-bound path; override via env var if needed
DATASET_DIR = Path(os.getenv("NEJM_DATASET_DIR", "/workspace/dataset"))
JSON_PATH = DATASET_DIR / "image_challenge_dataset_20231223.json"
IMAGES_DIR = DATASET_DIR / "images"


# --- Public API --------------------------------------------------------------

def load_nejm(n_samples: int = None, seed: int = 42, require_image: bool = True):
    """
    Load NEJM Image Challenge cases with rich clinical context.

    Args:
        n_samples: How many cases to return. If None, returns all available.
        seed:      Random seed for reproducible sampling. Use the SAME seed
                   across all communication modes so they evaluate on the
                   same questions.
        require_image: Skip cases whose image file is missing.

    Returns:
        List of dicts, each with keys:
            image:        PIL.Image (RGB)
            image_id:     int — the case's NEJM image id
            question:     str — full clinical vignette + diagnostic prompt
            options:      dict[str, str] — {"A": "...", "B": "...", ...}
            answer:       str — correct letter ("A".."E")
            brier_score:  float — per-case difficulty proxy (lower = easier)
    """
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"NEJM JSON not found at {JSON_PATH}. "
            "Set NEJM_DATASET_DIR or place the file at the default path."
        )

    with open(JSON_PATH) as f:
        all_cases = json.load(f)

    # Filter 1: cases flagged as having substantive clinical context.
    # The cx0 authors set this flag to identify the ~687 'good' cases — those
    # whose prompt contains presentation, history, labs, etc., rather than
    # generic queries like 'what is the diagnosis?'. Without this filter we'd
    # be back to VQA-RAD's problem: text agent has nothing to reason over.
    cases = [c for c in all_cases if c.get("relevant_context") == "yes"]

    # Filter 2: image must exist locally. A handful may have failed to download.
    if require_image:
        cases = [
            c for c in cases
            if (IMAGES_DIR / f"{c['image_id']:04d}.jpg").exists()
        ]

    # Reproducible sampling. The same seed yields the same case ordering,
    # so every communication mode evaluates on identical cases.
    rng = random.Random(seed)
    rng.shuffle(cases)

    if n_samples is not None:
        cases = cases[:n_samples]

    return [_to_example(c) for c in cases]

def load_nejm_split(split: str = "all", seed: int = 42, train_n: int = 350):
    """Load NEJM cases with train/test split matching ThoughtComm.

    Args:
        split: "all" (689 cases), "train" (first train_n), or "test" (remaining)
        seed: Must match ThoughtComm's seed (42) for alignment
        train_n: Must match ThoughtComm's TRAIN_SPLIT (350)

    Returns:
        List of example dicts, same format as load_nejm.
    """
    all_cases = load_nejm(n_samples=None, seed=seed)

    if split == "train":
        return all_cases[:train_n]
    elif split == "test":
        return all_cases[train_n:]
    else:
        return all_cases
    

# def image_to_base64(image: Image.Image) -> str:
#     """Convert a PIL image to a base64-encoded JPEG string for vision input."""
#     buffer = BytesIO()
#     image.save(buffer, format="JPEG", quality=95, subsampling=0)
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")

def jpeg_file_to_base64(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")

import re


def split_context_and_question(case_text: str) -> tuple[str, str]:
    """Split an NEJM case prompt into (clinical_context, diagnostic_question).

    Heuristic: the last sentence containing a question marker is the
    diagnostic question. Everything before it is clinical context.

    If no question marker is found, the final sentence is treated as the
    question and a warning is printed — this is rare but worth surfacing.

    Returns:
        (context, question) — both strings, both stripped.
        If splitting fails, context is empty and question is the whole input.
    """
    # Split into sentences. Naive split on sentence enders is fine here;
    # NEJM prompts don't contain abbreviations like "Dr." that would
    # confuse this. Keep the terminator with each sentence.
    sentences = re.split(r"(?<=[.?!])\s+", case_text.strip())

    if len(sentences) <= 1:
        # Single-sentence case — nothing to split. Treat whole thing as
        # question; text agent will have no context, which is fine.
        return "", case_text.strip()

    # Find the last sentence that looks like a question
    question_markers = ("?", "what ", "which ", "most likely",
                        "best ", "diagnosis", "cause")

    question_idx = None
    for i in range(len(sentences) - 1, -1, -1):
        lower = sentences[i].lower()
        if any(marker in lower for marker in question_markers):
            question_idx = i
            break

    if question_idx is None:
        # Fallback: just take the last sentence as the question.
        question_idx = len(sentences) - 1

    context = " ".join(sentences[:question_idx]).strip()
    question = " ".join(sentences[question_idx:]).strip()
    return context, question


def format_text_question(example: dict) -> str:
    """Text agent input: clinical context + diagnostic question + options. NO image.

    The text agent sees the full case — patient history, presentation, labs —
    and answers from clinical reasoning alone.
    """
    return f"""Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Based on the clinical history alone, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <your clinical reasoning in 2-3 sentences>"""


def format_vision_question(example: dict) -> str:
    """Vision agent input: diagnostic question + options + image only.

    The vision agent receives a referral-style prompt: the diagnostic
    question and options, plus the image, but WITHOUT patient history.
    This mirrors a radiology consult where the radiologist analyses the
    image in response to a specific query, without access to the chart.
    """
    _, diagnostic_question = split_context_and_question(example["question"])

    return f"""You are reviewing a medical image. A clinician has sent you the following query along with this image.

Query: {diagnostic_question}

Options:
{_format_options(example["options"])}

Examine the image and choose the most likely answer based on what you observe.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <what you observed in the image and how it informs your choice, 2-3 sentences>"""

# This was the previous input split, where the vision agent used to have the full text context

# def format_text_question(example: dict) -> str:
#     """Text agent input: clinical vignette + options. NO image."""
#     return f"""Clinical case:
# {example["question"]}

# Options:
# {_format_options(example["options"])}

# Based on the clinical history alone, choose the most likely diagnosis.
# Respond in this exact format:
# ANSWER: <single letter A-E>
# REASONING: <your clinical reasoning in 2-3 sentences>"""


# def format_vision_question(example: dict) -> str:
#     """Vision agent input: clinical vignette + options. Image passed separately."""
#     return f"""You are reviewing a medical image accompanying this case.

# Clinical case:
# {example["question"]}

# Options:
# {_format_options(example["options"])}

# Examine the image and choose the most likely diagnosis.
# Respond in this exact format:
# ANSWER: <single letter A-E>
# REASONING: <what you observed in the image and how it informs your choice, 2-3 sentences>"""

## This was the previous case where the meta agent used to have access to the clinical context 
# def format_meta_question(example: dict, text_output: str, vision_output: str) -> str:
#     """Meta agent input for OUTPUT-ONLY mode (Mode 1).

#     Future modes (CoT sharing, structured) will use different formatters here,
#     which is why this lives in the loader rather than being inlined in main.py.
#     """
#     return f"""You are a senior consultant reviewing two specialists' assessments of this case.

# Clinical case:
# {example["question"]}

# Options:
# {_format_options(example["options"])}

# Clinical Text Specialist's assessment:
# {text_output}

# Radiology Vision Specialist's assessment:
# {vision_output}

# Synthesise both assessments and choose the most likely diagnosis.
# Respond in this exact format:
# ANSWER: <single letter A-E>
# REASONING: <why you chose this answer, considering both specialists, 2-3 sentences>"""

def format_meta_question(example, text_output, vision_output):
    """Meta agent input for OUTPUT-ONLY mode (Mode 1).
    
    The meta agent sees ONLY the specialists' outputs and the answer options.
    It does NOT see the original clinical case — forcing it to synthesise
    from the specialists' reports rather than reasoning independently.
    """
    return f"""You are a senior consultant. Two specialists have assessed a clinical case.
You do not have access to the original case details — you must base your decision
entirely on the specialists' assessments below.

The available diagnostic options are:
{_format_options(example["options"])}

Clinical Text Specialist's assessment:
{text_output}

Radiology Vision Specialist's assessment:
{vision_output}

Based solely on these two assessments, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you chose this answer based on the specialists' input, 2-3 sentences>"""

def format_single_agent_question(example: dict) -> str:
    """Single-agent baseline: one model sees clinical text, options, AND image.
    
    This is the strong-baseline condition — same model as the multi-agent
    pipeline, but given all information at once with no decomposition.
    """
    return f"""You are reviewing a medical case with an accompanying image.

Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Examine both the clinical history and the image, then choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <how the clinical history and image findings together support your choice, 2-3 sentences>"""

def get_ground_truth(example: dict) -> str:
    """Returns the correct answer letter (A-E)."""
    return example["answer"]

def format_text_revision(example, text_r1_output, vision_r1_output):
    """Round 2: Text agent revises after seeing vision specialist's reasoning.

    Exchange-of-Thought style — the text agent sees the vision specialist's
    full R1 output and decides whether to update its own answer.
    """
    _, diagnostic_question = split_context_and_question(example["question"])

    return f"""You previously assessed a clinical case and gave your answer.
Now you are shown the radiology vision specialist's independent assessment
of the same case. They examined the medical image for this query:
"{diagnostic_question}"

Your original assessment:
{text_r1_output}

Vision Specialist's assessment:
{vision_r1_output}

Consider whether the vision specialist's findings change your diagnosis.
You may keep your original answer or revise it.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you kept or changed your answer, 2-3 sentences>"""


def format_vision_revision(example, vision_r1_output, text_r1_output):
    """Round 2: Vision agent revises after seeing text specialist's reasoning.

    The vision agent sees the text specialist's clinical reasoning and
    decides whether to update its image-based assessment.
    """
    return f"""You previously examined a medical image and gave your assessment.
Now you are shown the clinical text specialist's independent assessment
of the same case. They had access to the full patient history.

Your original assessment:
{vision_r1_output}

Clinical Text Specialist's assessment:
{text_r1_output}

Consider whether the text specialist's clinical reasoning changes your diagnosis.
You may keep your original answer or revise it.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you kept or changed your answer, 2-3 sentences>"""


def format_meta_question_debate(example, text_r2_output, vision_r2_output):
    """Meta agent input for Mode 2b (debate).

    Sees ONLY the revised (R2) outputs from both specialists plus options.
    Same restriction as Mode 1: no access to original clinical case.
    """
    return f"""You are a senior consultant. Two specialists have assessed a clinical case
through two rounds — first independently, then after reviewing each other's reasoning.
You are seeing their final revised assessments.

You do not have access to the original case details — you must base your decision
entirely on the specialists' revised assessments below.

The available diagnostic options are:
{_format_options(example["options"])}

Clinical Text Specialist's revised assessment:
{text_r2_output}

Radiology Vision Specialist's revised assessment:
{vision_r2_output}

Based solely on these revised assessments, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you chose this answer based on the specialists' revised input, 2-3 sentences>"""
# --- Internals ---------------------------------------------------------------

def _to_example(case: dict) -> dict:
    """Convert a raw NEJM JSON case to the pipeline's example format."""
    image_path = IMAGES_DIR / f"{case['image_id']:04d}.jpg"
    # .convert('RGB') normalises mode — some NEJM images are CMYK or grayscale,
    # which break downstream JPEG re-encoding for the vision agent.
    image = Image.open(image_path).convert("RGB")

    options = {}
    for letter in ("A", "B", "C", "D", "E"):
        text = case.get(f"option_{letter}")
        if text:
            options[letter] = text

    return {
        "image": image,
        "image_id": case["image_id"],
        "question": case["question"],
        "options": options,
        "answer": case["correct_answer"].strip().upper(),
        "brier_score": float(case.get("brier_score", 0.0)),
    }


def _format_options(options: dict) -> str:
    return "\n".join(f"  {letter}) {text}" for letter, text in options.items())

## for mode-3 structured JSON ouput

"""
STRUCTURED COMMUNICATION FORMATTERS (Mode 3)

Add these functions to data/loader.py alongside the existing formatters.
Also add this import at the top of loader.py:

    from evaluation.json_parser import parse_specialist_json

These formatters produce structured JSON output from specialists,
which the meta agent receives as parsed, typed data rather than
free-text reasoning.
"""


def format_text_question_structured(example: dict) -> str:
    """Text agent structured prompt: produce JSON instead of free text.

    Same information access as the standard text agent (full clinical
    context + options, no image), but output format is structured JSON.
    """
    return f"""You are an experienced clinical physician.
You will be given a clinical case description and multiple-choice options, without any image.
Reason from clinical knowledge to choose the most likely diagnosis.

Respond with ONLY a valid JSON object. No other text, no markdown backticks.

{{
  "answer": "<single letter A-E>",
  "confidence": <number between 0.0 and 1.0>,
  "key_findings": ["<finding 1>", "<finding 2>"],
  "supporting_evidence": "<one sentence explaining your choice>",
  "alternative_considered": "<single letter A-E of your second choice>",
  "why_not_alternative": "<one sentence why you rejected it>"
}}

Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}"""


def format_vision_question_structured(example: dict) -> str:
    """Vision agent structured prompt: produce JSON from image analysis.

    Same information access as the standard vision agent (diagnostic
    question + options + image, no patient history), structured output.
    """
    _, diagnostic_question = split_context_and_question(example["question"])

    return f"""You are an experienced radiologist.
You are reviewing a medical image. A clinician sent you the following query.

Respond with ONLY a valid JSON object. No other text, no markdown backticks.

{{
  "answer": "<single letter A-E>",
  "confidence": <number between 0.0 and 1.0>,
  "key_findings": ["<visual finding 1>", "<visual finding 2>"],
  "supporting_evidence": "<one sentence about what you see in the image>",
  "alternative_considered": "<single letter A-E of your second choice>",
  "why_not_alternative": "<one sentence why you rejected it>"
}}

Query: {diagnostic_question}

Options:
{_format_options(example["options"])}"""


def format_meta_question_structured(example: dict, text_json: dict, vision_json: dict):
    """Meta agent for Mode 3: receives structured specialist assessments.

    Constrained to choose ONLY from the specialists' proposed answers.
    Confidence scores give the meta agent a principled basis for
    arbitrating between disagreeing specialists.
    """
    # Build constrained option set from specialists' answers
    text_ans = text_json.get("answer", "?")
    vision_ans = vision_json.get("answer", "?")
    proposed = set()
    if text_ans in example["options"]:
        proposed.add(text_ans)
    if vision_ans in example["options"]:
        proposed.add(vision_ans)

    # Fallback: if both failed to produce valid answers, show all options
    if not proposed:
        proposed_options = _format_options(example["options"])
        constraint_note = "Neither specialist produced a clear answer. Choose from all available options."
    else:
        proposed_options = "\n".join(
            f"  {k}) {example['options'][k]}" for k in sorted(proposed)
        )
        constraint_note = "You must choose one of the options proposed by the specialists."

    return f"""You are a senior consultant reviewing structured assessments from two specialists.
You do not have access to the original case details. {constraint_note}

The specialists proposed these options:
{proposed_options}

── Clinical Text Specialist ──
Answer: {text_ans}
Confidence: {text_json.get('confidence', 'N/A')}
Key findings: {', '.join(text_json.get('key_findings', [])) or 'None provided'}
Supporting evidence: {text_json.get('supporting_evidence', 'None provided')}
Alternative considered: {text_json.get('alternative_considered', 'N/A')}
Why rejected: {text_json.get('why_not_alternative', 'N/A')}

── Radiology Vision Specialist ──
Answer: {vision_ans}
Confidence: {vision_json.get('confidence', 'N/A')}
Key findings: {', '.join(vision_json.get('key_findings', [])) or 'None provided'}
Supporting evidence: {vision_json.get('supporting_evidence', 'None provided')}
Alternative considered: {vision_json.get('alternative_considered', 'N/A')}
Why rejected: {vision_json.get('why_not_alternative', 'N/A')}

Based on the specialists' structured assessments and confidence levels, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter from the proposed options above>
REASONING: <why you chose this, referencing the specialists' confidence and findings, 2-3 sentences>"""
