"""
State for the conditional-debate LangGraph experiment.

Main idea:
  1. Text agent gives an initial answer from vignette/options only.
  2. Vision gate agent evaluates the image against the options only.
  3. Agreement checker decides whether debate is needed.
  4. If confident agreement, skip debate.
  5. If disagreement/uncertainty, run directed visual consultation.
"""

from typing import TypedDict, Any


class ConditionalDebateState(TypedDict, total=False):
    # Inputs
    image_id: int
    question: str
    options: dict
    ground_truth: str          # logging only, never prompted
    image_b64: str

    # Text initial / evolving answer
    text_answer: str
    text_answer_initial: str
    text_confidence: int       # 1-5 self-rated confidence
    text_assessment: str

    # Vision gate output: image + options, no clinical vignette
    image_description: str
    vision_answer: str         # A-E or UNKNOWN
    vision_confidence: int     # 1-5 self-rated confidence
    vision_reasoning: str
    visual_support: dict       # option -> supports/refutes/not_assessable/etc.

    # Conditional routing fields
    agents_agreed: bool
    debate_triggered: bool
    routing_reason: str
    final_mode: str            # direct_agreement, direct_text_high_conf, debated

    # Debate transcript
    visual_questions: list     # [[q1, q2], [q3], ...]
    visual_answers: list       # [[answer_text], [answer_text], ...]
    round: int
    done_debating: bool

    # Final
    final_output: str
    final_answer: str
