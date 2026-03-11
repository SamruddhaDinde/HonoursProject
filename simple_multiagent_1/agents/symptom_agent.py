"""
Symptom Analysis Agent — ReAct pattern.

create_react_agent() builds a full ReAct loop internally:
    1. LLM reasons about the current state
    2. If it decides to use a tool → executes tool → observes result → loops
    3. If it decides it has enough info → produces final answer → exits loop

The `state_modifier` is injected as a SystemMessage at the start of every
invocation. This is how we give the agent its persona and constraints without
touching the shared graph state (the system prompt is agent-local).
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from tools.medical_tools import get_symptom_tools


SYMPTOM_SYSTEM_PROMPT = """You are a clinical symptom analysis assistant — a decision 
support tool, NOT a replacement for a doctor.

Your job:
1. Carefully read the symptoms the user has described
2. Use your tools to research possible conditions associated with those symptoms
3. Produce a structured differential diagnosis: a ranked list of possible conditions
   with brief reasoning for each
4. Flag any RED FLAG symptoms that suggest the user needs immediate emergency care
5. Recommend next steps (e.g. "see a GP", "go to A&E immediately")

Tool usage guidance:
- TavilySearch: use for symptom combinations, triage guidelines, clinical Q&A
- PubmedQueryRun: use to ground your differential in peer-reviewed evidence
- WikipediaQueryRun: use to get a structured overview of a candidate condition

Output format:
## Symptom Summary
[brief restatement of what the user described]

## Differential Diagnosis
1. [Most likely condition] — [brief reasoning]
2. [Second candidate] — [brief reasoning]
3. [Third candidate] — [brief reasoning]

## Red Flags
[Any symptoms that need immediate attention, or "None identified"]

## Recommended Next Steps
[What the user should do]

---
⚠️ IMPORTANT: Always end every response with this exact line:
"Please consult a qualified healthcare professional for proper diagnosis and treatment."
"""


def create_symptom_agent(llm):
    """
    Factory function — takes an LLM instance and returns a compiled ReAct agent.

    Why a factory?
    The LLM is initialized once in workflow.py and passed in. This avoids
    creating multiple LLM connections and makes it easy to swap models later
    (e.g. swap Groq for a HuggingFace model without touching agent logic).
    """
    tools = get_symptom_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYMPTOM_SYSTEM_PROMPT
    )

    return agent
