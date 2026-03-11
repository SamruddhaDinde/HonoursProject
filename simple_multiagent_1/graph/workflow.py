"""
Supervisor + Graph Assembly.

This is the heart of the multi-agent system.

Architecture recap:
    START
      ↓
  [supervisor]  ← reads state, decides who goes next
      ↓ (conditional edge reads state["next"])
   ┌──┴──────────────┐
   ↓                 ↓
[symptom_agent]  [research_agent]
   ↓                 ↓
   └──────┬──────────┘
          ↓
      [supervisor]  ← re-evaluates. Is the answer complete? → END
                                    Need more depth?        → route again

Key design decision — why route back to supervisor after each agent?
Because the supervisor needs to re-read the updated state (which now includes
the agent's response) to decide if the job is done. This is the "evaluate
then terminate" pattern. Without it, agents would always run exactly once.
"""

from typing import Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents.symptom_agent import create_symptom_agent
from agents.research_agent import create_research_agent


# ---------------------------------------------------------------------------
# Routing Schema
# ---------------------------------------------------------------------------
# We use Pydantic + structured output to force the supervisor LLM to return
# a valid routing decision — not free text.
#
# Why structured output instead of parsing free text?
# Reliability. If we asked the LLM "who should handle this?" in plain text,
# we'd need to parse its response and handle edge cases. with_structured_output()
# forces the model to return exactly the fields we specify, with the exact
# Literal values we defined. Routing failures become near-impossible.

class RouteDecision(BaseModel):
    next: Literal["symptom_agent", "research_agent", "FINISH"]
    reasoning: str  # makes the supervisor's decision transparent / debuggable


# ---------------------------------------------------------------------------
# Supervisor System Prompt
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM_PROMPT = """You are a medical query supervisor. Your only job is to
read the conversation and decide which specialist agent should act next.

Available agents:
- symptom_agent:   Handles queries where the user describes symptoms and wants
                   to know what condition they might have (differential diagnosis)
- research_agent:  Handles queries about specific medical conditions, treatments,
                   drugs, or clinical guidelines
- FINISH:          Use when a specialist agent has already provided a complete,
                   thorough answer and no further action is needed

Routing rules (follow strictly):
1. If no specialist has responded yet → route based on query type (symptoms → symptom_agent,
   specific medical topic → research_agent)
2. If ANY specialist agent has already produced a response → choose FINISH unless
   the user's query has a clearly distinct second aspect that the OTHER agent must handle
3. If both agents have responded (in any order) → always choose FINISH
4. Never route to the same agent twice

Default to FINISH. Only route to a second agent when it is clearly necessary.

Always fill in `reasoning` — this is your chain-of-thought and is critical for debugging."""


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assembles and compiles the full multi-agent graph.

    Returns a compiled LangGraph graph ready to be invoked.
    Call this once at startup — compilation is expensive, invocation is cheap.
    """

    # Single LLM instance shared across all agents.
    # temperature=0 for the supervisor: we want deterministic routing decisions.
    # The specialist agents inherit this — you can give them their own temperature
    # later if you want more creative/varied responses.
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Supervisor gets structured output forcing — it MUST return a RouteDecision.
    # The underlying LLM is the same, but with an output parser wrapping it.
    supervisor_llm = llm.with_structured_output(RouteDecision)

    # Build the specialist agents (each is a compiled ReAct subgraph internally)
    symptom_agent = create_symptom_agent(llm)
    research_agent = create_research_agent(llm)

    # -----------------------------------------------------------------------
    # Node Definitions
    # -----------------------------------------------------------------------

    MAX_AGENT_CALLS = 2  # hard cap: at most 2 specialist agent invocations per query

    def supervisor_node(state: AgentState) -> dict:
        """
        Reads the full conversation history and decides routing.

        Why prepend SystemMessage separately instead of using state_modifier?
        The supervisor doesn't use create_react_agent — it calls the LLM directly.
        So we manually build the message list: [system prompt] + [all messages so far].

        The supervisor sees the FULL history, including agents' previous responses.
        This is how it knows whether the job is done (FINISH) or needs more work.
        """
        # Hard cap: if we've already hit the max, force FINISH regardless of LLM decision.
        if state.get("agent_calls", 0) >= MAX_AGENT_CALLS:
            print(f"\n{'='*50}")
            print(f"[Supervisor] → Routing to: FINISH (max agent calls reached)")
            print(f"{'='*50}\n")
            return {"next": "FINISH"}

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            *state["messages"]
        ]

        decision = supervisor_llm.invoke(messages)

        # Code-level guard: never route to the same agent that just ran.
        # The LLM sometimes ignores the prompt rule, so we enforce it here.
        next_node = decision.next
        if next_node == state.get("last_agent", ""):
            next_node = "FINISH"

        print(f"\n{'='*50}")
        print(f"[Supervisor] → Routing to: {next_node}")
        print(f"[Supervisor] Reasoning: {decision.reasoning}")
        print(f"{'='*50}\n")

        # We only update `next` — messages are untouched by the supervisor.
        # The supervisor observes, it doesn't talk.
        return {"next": next_node}

    def symptom_node(state: AgentState) -> dict:
        """
        Invokes the symptom ReAct agent with the current state.

        The ReAct agent runs its internal loop here:
        - It receives the full message history via `state`
        - It reasons, calls tools, observes results, reasons again
        - When satisfied, it returns its final message

        We return only {"messages": ...} — LangGraph's add_messages reducer
        appends the agent's new messages to the existing history automatically.
        We do NOT overwrite state["next"] here — that's the supervisor's job.
        """
        print("[Symptom Agent] Running analysis...\n")
        result = symptom_agent.invoke(state)
        return {
            "messages": result["messages"],
            "agent_calls": state.get("agent_calls", 0) + 1,
            "last_agent": "symptom_agent"
        }

    def research_node(state: AgentState) -> dict:
        """
        Invokes the research ReAct agent. Same pattern as symptom_node.

        Keeping the node functions thin like this (just invoke + return messages)
        is intentional. All agent-specific logic lives in the agent modules.
        """
        print("[Research Agent] Searching literature...\n")
        result = research_agent.invoke(state)
        return {
            "messages": result["messages"],
            "agent_calls": state.get("agent_calls", 0) + 1,
            "last_agent": "research_agent"
        }

    def route_after_supervisor(state: AgentState) -> str:
        """
        The routing function for the conditional edge.

        This function is called by LangGraph after the supervisor node runs.
        It reads state["next"] (which the supervisor just wrote) and returns
        it as a string — LangGraph uses this to look up the next node in the
        edges dict defined in add_conditional_edges().

        Why a separate function instead of a lambda?
        Clarity. A named function is easier to read, test, and debug.
        """
        return state["next"]

    # -----------------------------------------------------------------------
    # Graph Assembly
    # -----------------------------------------------------------------------

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("symptom_agent", symptom_node)
    builder.add_node("research_agent", research_node)

    # Entry point: every invocation starts at the supervisor
    builder.add_edge(START, "supervisor")

    # The supervisor's output drives a conditional edge.
    # The dict maps: routing function return value → node name to visit.
    # "FINISH" maps to the built-in END sentinel which terminates the graph.
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "symptom_agent": "symptom_agent",
            "research_agent": "research_agent",
            "FINISH": END
        }
    )

    # After each specialist finishes, unconditionally return to supervisor.
    # The supervisor will re-evaluate and either route again or FINISH.
    builder.add_edge("symptom_agent", "supervisor")
    builder.add_edge("research_agent", "supervisor")

    # compile() validates the graph structure and returns an executable object.
    # If you have disconnected nodes or missing edges, this will raise an error.
    return builder.compile()
