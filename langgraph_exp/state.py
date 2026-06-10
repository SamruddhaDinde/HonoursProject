"""
Shared state for the orchestrator LangGraph workflow.

The state travels through every node. Each node reads what it needs and
returns a partial dict that LangGraph merges in. Keeping ALL intermediate
fields here (not just the final answer) is what makes per-case analysis and
LangSmith traces useful later — nothing is thrown away mid-pipeline.
"""

from typing import TypedDict, Optional


class OrchestratorState(TypedDict, total=False):
    # ---- Inputs (populated before the graph runs) ----
    image_id: int
    question: str                 # full clinical vignette + diagnostic question
    options: dict                 # {"A": "...", ...}
    ground_truth: str             # gold letter — carried for logging ONLY, never shown to nodes
    image_b64: str                # base64 JPEG for the vision node

    # ---- Vision describe node output ----
    image_description: str        # pure description, NO answer, NO question seen

    # ---- Text search node output ----
    text_assessment: str          # the text agent's reasoning + tentative answer
    text_answer: str              # extracted letter (A-E or UNKNOWN)
    search_queries: list          # every query the agent issued (audit trail)
    search_snippets: list         # every snippet returned (contamination audit)

    # ---- Orchestrator final output ----
    final_output: str             # orchestrator's full text
    final_answer: str             # extracted letter (A-E or UNKNOWN)

    # ---- Config (per-run, read by nodes/edges) ----
    use_web_search: bool          # toggles the search tool for the paired control