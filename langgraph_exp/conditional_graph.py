"""
Conditional-debate graph.

Flow:
  START -+-> text_initial ----+
         +-> vision_gate -----+-> agreement_check -+-> direct_final -> END
                                                   +-> vision_query -> text_revise -> ... -> orchestrator -> END

The graph only debates when text and vision disagree, or when confidence is too
low for a direct decision.
"""

from langgraph.graph import StateGraph, START, END

from .conditional_state import ConditionalDebateState
from .conditional_nodes import (
    text_initial_node,
    vision_gate_node,
    agreement_check_node,
    direct_final_node,
    vision_query_node,
    text_revise_node,
    orchestrator_node,
)


def _route_after_agreement_check(state):
    if state.get("debate_triggered"):
        return "vision_query"
    return "direct_final"


def _route_after_text_revise(state):
    if state.get("done_debating"):
        return "orchestrator"
    return "vision_query"


def build_conditional_debate_graph():
    g = StateGraph(ConditionalDebateState)

    g.add_node("text_initial", text_initial_node)
    g.add_node("vision_gate", vision_gate_node)
    g.add_node("agreement_check", agreement_check_node)
    g.add_node("direct_final", direct_final_node)
    g.add_node("vision_query", vision_query_node)
    g.add_node("text_revise", text_revise_node)
    g.add_node("orchestrator", orchestrator_node)

    # Parallel initial assessment: text from vignette/options, vision from image/options.
    g.add_edge(START, "text_initial")
    g.add_edge(START, "vision_gate")

    # Fan-in. LangGraph waits for both predecessors before agreement_check.
    g.add_edge("text_initial", "agreement_check")
    g.add_edge("vision_gate", "agreement_check")

    # Decide whether to skip or run debate.
    g.add_conditional_edges(
        "agreement_check",
        _route_after_agreement_check,
        {"direct_final": "direct_final", "vision_query": "vision_query"},
    )

    # Debate loop, only for routed cases.
    g.add_edge("vision_query", "text_revise")
    g.add_conditional_edges(
        "text_revise",
        _route_after_text_revise,
        {"vision_query": "vision_query", "orchestrator": "orchestrator"},
    )

    g.add_edge("direct_final", END)
    g.add_edge("orchestrator", END)

    return g.compile()
