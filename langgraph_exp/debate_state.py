"""
State for the directed-debate orchestrator.

Flow:
  text_initial  -> (asks visual questions) -> vision_query -> text_revise
                -> [loop up to MAX_ROUNDS, or stop if no questions] -> orchestrator

Design rationale (from prior results):
  * ASYMMETRIC: only the text agent revises. Symmetric debate previously
    caused sycophantic convergence that dragged the strong text agent down
    (54%->43% in Mode 3b). Here vision never revises and never sees options.
  * DIRECTED: the text agent asks SPECIFIC visual questions rather than
    reacting to a vision diagnosis. Vision answers only those questions.
  * Every round's questions/answers are retained for trace analysis — the
    point of this experiment is to SEE whether directed visual queries change
    the text agent's answer, not just the final accuracy.
"""

from typing import TypedDict


class DebateState(TypedDict, total=False):
    # Inputs
    image_id: int
    question: str
    options: dict
    ground_truth: str          # logging only, never prompted
    image_b64: str

    # Text agent, evolving
    text_answer: str           # current letter
    text_assessment: str       # current full reasoning
    text_answer_initial: str   # answer BEFORE seeing any visual answers (key for analysis)

    # Debate transcript (lists, one entry per round)
    visual_questions: list     # [[q1, q2], [q3], ...] per round
    visual_answers: list       # [[a1, a2], [a3], ...] per round

    # Loop control
    round: int                 # current round index
    done_debating: bool        # set when text agent asks no further questions

    # Final
    image_description: str      # baseline full description (round 0, for orchestrator)
    final_output: str
    final_answer: str