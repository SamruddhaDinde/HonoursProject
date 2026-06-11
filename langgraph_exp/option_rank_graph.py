"""
Graph wiring for the option-ranking fusion experiment.

    START ─┬─> text_rank ───┐
           └─> vision_rank ──┤
                             └─> fusion ─> END

The two ranking nodes are independent and write disjoint keys, so they can run
from START in the same LangGraph superstep. With a single Ollama backend the GPU
may still serialize the actual model calls, but the graph structure is correct.
"""

from langgraph.graph import StateGraph, START, END

from .option_rank_state import OptionRankState
from .option_rank_nodes import text_rank_node, vision_rank_node, fusion_node


def build_option_rank_graph():
    g = StateGraph(OptionRankState)

    g.add_node("text_rank", text_rank_node)
    g.add_node("vision_rank", vision_rank_node)
    g.add_node("fusion", fusion_node)

    g.add_edge(START, "text_rank")
    g.add_edge(START, "vision_rank")

    # LangGraph waits until both predecessors have written their state updates.
    g.add_edge("text_rank", "fusion")
    g.add_edge("vision_rank", "fusion")

    g.add_edge("fusion", END)
    return g.compile()
