"""
State for the conservative conditional-debate LangGraph experiment.

Purpose:
  Test whether multimodal debate can help without allowing ambiguous visual
  evidence to drag down a stronger text-only diagnosis.

Policy:
  - Text initial node sees clinical vignette + options, but no image.
  - Vision gate sees image only, no vignette and no answer options.
  - Vision query sees image + text-generated visual questions only.
  - Text revise sees vignette/options + radiologist answers, but no raw image.
  - Final decision is deterministic: the guarded text answer is used directly.
"""

from typing import TypedDict


class ConservativeDebateState(TypedDict, total=False):
    # Inputs
    image_id: int
    question: str
    options: dict
    ground_truth: str          # logging only; never prompted
    image_b64: str

    # Initial text-only answer
    text_answer_initial: str
    text_initial_confidence: int
    text_initial_assessment: str

    # Current guarded text answer after debate/revision
    text_answer: str
    text_confidence: int
    text_assessment: str

    # Vision gate: image-only assessment, no options/vignette
    image_description: str
    image_diagnosticity: int   # 1-5; whether image seems diagnostically useful
    image_uncertainty: str
    image_gate_reasoning: str

    # Conditional routing
    debate_triggered: bool
    final_mode: str            # direct_text_high_conf, direct_image_low_value, debated_guarded
    routing_reason: str

    # Debate transcript
    visual_questions: list     # [[q1, q2], [q3], ...]
    visual_answers: list       # [[answer_text], [answer_text], ...]
    round: int
    done_debating: bool

    # Revision proposal + conservative guard audit
    proposed_text_answer: str
    proposed_text_confidence: int
    proposed_text_assessment: str
    visual_evidence_strength: str      # none/weak/moderate/strong
    visual_contradicts_previous: bool
    visual_supports_new_answer: bool
    revision_allowed: bool
    revision_guard_reason: str
    guard_blocked_change: bool

    # Final
    final_output: str
    final_answer: str
