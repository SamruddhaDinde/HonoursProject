"""
State for the option-ranking fusion LangGraph experiment.

Purpose:
  Compare against single-answer fusion/debate by making each modality evaluate
  every answer option before final fusion.

Distributed-input policy:
  * text_rank_node sees clinical vignette + options, but NO image.
  * vision_rank_node sees image + options, but NO clinical vignette.
  * fusion_node sees the vignette, options, text ranking, and vision ranking,
    but NO raw image.

This is an OPTION-AWARE visual condition: the vision model sees the five answer
choices so it can say which labels are visually supported/refuted/not assessable.
Do not describe this as a pure image-only condition.
"""

from typing import TypedDict, Optional, Dict, Any


class OptionRankState(TypedDict, total=False):
    # Inputs
    image_id: int
    question: str
    options: dict
    ground_truth: str          # logging only; never placed in prompts
    image_b64: str

    # Text ranking output: vignette + options, no image
    text_raw_output: str
    text_top_answer: str       # A-E; forced fallback if parsing fails
    text_confidence: int       # 1-5
    text_scores: Dict[str, int]  # {"A": 1-5, ...}
    text_reasoning: str
    text_parse_failed: bool

    # Vision ranking output: image + options, no vignette
    vision_raw_output: str
    vision_top_answer: str       # A-E or UNKNOWN
    vision_confidence: int       # 0-5, can be 0 if not assessable
    vision_scores: Dict[str, int]  # {"A": 0-5, ...}
    vision_support: Dict[str, str] # supports/weak_support/neutral/refutes/not_assessable
    key_visual_findings: str
    vision_reasoning: str
    vision_parse_failed: bool

    # Fusion output
    final_output: str
    final_answer: str          # A-E; forced fallback if parsing fails
    fusion_changed_from_text: bool
