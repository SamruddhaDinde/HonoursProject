"""
Graph wiring for the orchestrator workflow.

    START ─> vision_describe ─┐
            ─> text_search ───┤  (these two run, then both feed)
                              └─> orchestrator ─> END

Vision and text are independent given the inputs, so they could run in
parallel. With a single Ollama instance serving both models the calls
serialize at the GPU anyway, so we keep a simple sequential wiring
(vision -> text -> orchestrator) for predictable ordering in traces.
If you later serve the two models on separate endpoints, switch to the
parallel fan-out noted below.
"""

from langgraph.graph import StateGraph, START, END

from .state import OrchestratorState
from .nodes import vision_describe_node, text_search_node, orchestrator_node


def build_graph():
    g = StateGraph(OrchestratorState)

    g.add_node("vision_describe", vision_describe_node)
    g.add_node("text_search", text_search_node)
    g.add_node("orchestrator", orchestrator_node)

    # Sequential: vision, then text, then orchestrator.
    # g.add_edge(START, "vision_describe")
    # g.add_edge("vision_describe", "text_search")
    # g.add_edge("text_search", "orchestrator")
    # g.add_edge("orchestrator", END)

    # --- Parallel alternative (use only with two separate model endpoints) ---
    g.add_edge(START, "vision_describe")
    g.add_edge(START, "text_search")
    g.add_edge("vision_describe", "orchestrator")
    g.add_edge("text_search", "orchestrator")
    #LangGraph waits for both predecessors before running orchestrator.

    return g.compile()