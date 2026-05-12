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


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string for vision input."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def format_text_question(example: dict) -> str:
    """Text agent input: clinical vignette + options. NO image."""
    return f"""Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Based on the clinical history alone, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <your clinical reasoning in 2-3 sentences>"""


def format_vision_question(example: dict) -> str:
    """Vision agent input: clinical vignette + options. Image passed separately."""
    return f"""You are reviewing a medical image accompanying this case.

Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Examine the image and choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <what you observed in the image and how it informs your choice, 2-3 sentences>"""


def format_meta_question(example: dict, text_output: str, vision_output: str) -> str:
    """Meta agent input for OUTPUT-ONLY mode (Mode 1).

    Future modes (CoT sharing, structured) will use different formatters here,
    which is why this lives in the loader rather than being inlined in main.py.
    """
    return f"""You are a senior consultant reviewing two specialists' assessments of this case.

Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Clinical Text Specialist's assessment:
{text_output}

Radiology Vision Specialist's assessment:
{vision_output}

Synthesise both assessments and choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you chose this answer, considering both specialists, 2-3 sentences>"""

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


