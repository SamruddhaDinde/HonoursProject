"""
Conservative conditional-debate graph.

Flow:
  START -+-> text_initial ----+
         +-> vision_gate -----+-> conservative_route
                                  +-> final_decision -> END
                                  +-> vision_query -> text_revise -> ... -> final_decision -> END

Unlike the earlier conditional debate, final_decision is deterministic and uses
only the guarded text answer. This intentionally avoids a final orchestrator
from over-weighting weak visual evidence.
"""

from langgraph.graph import StateGraph, START, END

from .conservative_state import ConservativeDebateState
from .conservative_nodes import (
    text_initial_node,
    vision_gate_node,
    conservative_route_node,
    vision_query_node,
    text_revise_node,
    final_decision_node,
)


def _route_after_conservative_route(state):
    if state.get("debate_triggered"):
        return "vision_query"
    return "final_decision"


def _route_after_text_revise(state):
    if state.get("done_debating"):
        return "final_decision"
    return "vision_query"


def build_conservative_debate_graph():
    g = StateGraph(ConservativeDebateState)

    g.add_node("text_initial", text_initial_node)
    g.add_node("vision_gate", vision_gate_node)
    g.add_node("conservative_route", conservative_route_node)
    g.add_node("vision_query", vision_query_node)
    g.add_node("text_revise", text_revise_node)
    g.add_node("final_decision", final_decision_node)

    # Parallel initial assessment.
    g.add_edge(START, "text_initial")
    g.add_edge(START, "vision_gate")

    # Fan-in. LangGraph waits for both predecessors before routing.
    g.add_edge("text_initial", "conservative_route")
    g.add_edge("vision_gate", "conservative_route")

    g.add_conditional_edges(
        "conservative_route",
        _route_after_conservative_route,
        {"vision_query": "vision_query", "final_decision": "final_decision"},
    )

    g.add_edge("vision_query", "text_revise")
    g.add_conditional_edges(
        "text_revise",
        _route_after_text_revise,
        {"vision_query": "vision_query", "final_decision": "final_decision"},
    )

    g.add_edge("final_decision", END)
    return g.compile()
