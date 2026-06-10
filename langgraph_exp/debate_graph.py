"""
Directed-debate graph with PARALLEL initialisation.

  START -+-> vision_describe -+
         +-> text_initial ----+-> join -[route]-> vision_query -> text_revise -[route]- ...
                                                       +--------> orchestrator -> END

Parallelism: vision_describe (baseline full description) and text_initial
(answer + visual questions) are independent, so both run from START in the same
superstep. They write DISJOINT state keys -- vision_describe writes only
image_description; text_initial owns all loop-control keys (round,
visual_questions, visual_answers, done_debating) -- so there is no
concurrent-update conflict.

`join` is a no-op barrier: LangGraph runs it only after BOTH predecessors
complete, giving a clean fan-in. The debate loop downstream is sequential
because each vision_query depends on the preceding text turn.

The single-Ollama backend serialises the two initial calls at the GPU, so the
wall-clock saving is the overlap of orchestration/IO, not full concurrency;
the structure is still correct and matches the requested topology.
"""

from langgraph.graph import StateGraph, START, END

from .debate_state import DebateState
from .debate_nodes import (
    vision_describe_node, text_initial_node,
    vision_query_node, text_revise_node, orchestrator_node,
)


def join_node(state):
    """Barrier: runs only after both parallel initial branches finish.
    Returns no updates -- it exists purely to fan-in before the debate loop."""
    return {}


def _route_after_text(state):
    """Continue debating or go to final fusion.

    Enforces a MINIMUM of one debate round: at round 0 we always proceed to
    vision_query. After round >= 1, stop when the text agent asks nothing
    further or MAX_ROUNDS is hit. Guarantees every case gets >=1 consultation,
    so the run is a clean test of debate rather than a mix of debated and
    non-debated cases."""
    if state.get("round", 0) < 1:
        return "vision_query"
    if state.get("done_debating"):
        return "orchestrator"
    return "vision_query"


def build_debate_graph():
    g = StateGraph(DebateState)

    g.add_node("vision_describe", vision_describe_node)
    g.add_node("text_initial", text_initial_node)
    g.add_node("join", join_node)
    g.add_node("vision_query", vision_query_node)
    g.add_node("text_revise", text_revise_node)
    g.add_node("orchestrator", orchestrator_node)

    # Parallel fan-out: both initial nodes start together.
    g.add_edge(START, "vision_describe")
    g.add_edge(START, "text_initial")

    # Fan-in barrier: join runs only after BOTH complete.
    g.add_edge("vision_describe", "join")
    g.add_edge("text_initial", "join")

    # From the join, branch into the (sequential) debate loop or straight to fusion.
    g.add_conditional_edges("join", _route_after_text,
                            {"vision_query": "vision_query", "orchestrator": "orchestrator"})

    # Sequential debate cycle.
    g.add_edge("vision_query", "text_revise")
    g.add_conditional_edges("text_revise", _route_after_text,
                            {"vision_query": "vision_query", "orchestrator": "orchestrator"})

    g.add_edge("orchestrator", END)
    return g.compile()