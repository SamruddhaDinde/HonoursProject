from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The single shared object that flows through every node in the graph.

    Why TypedDict?
    LangGraph requires state to be a dict-like structure so it can
    merge partial updates from each node. TypedDict gives us type
    safety on top of that.

    Fields:
        messages:    Full conversation history. The `add_messages` reducer
                     APPENDS new messages rather than overwriting — this is
                     what gives agents memory of what happened before them.

        next:        Written by the supervisor node. Tells the conditional
                     edge which specialist agent (or END) to visit next.
                     This is the routing signal for the entire graph.

        agent_calls: Counts how many specialist agents have been invoked
                     this turn. Used to enforce a hard cap and prevent
                     infinite loops when the supervisor fails to FINISH.
    """
    messages: Annotated[list, add_messages]
    next: str
    agent_calls: int
    last_agent: str  # last specialist that ran; prevents back-to-back same-agent routing
