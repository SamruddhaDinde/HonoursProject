"""
Tool factories for each specialist agent.

Design principle: each agent gets only the tools it actually needs.
Giving every agent every tool wastes tokens (the tool schemas are injected
into the prompt) and makes routing decisions noisier.

Tools used:
- TavilySearch       → real-time web search. Needs TAVILY_API_KEY in .env
- PubmedQueryRun     → searches PubMed medical literature. FREE, no key needed.
- WikipediaQueryRun  → general condition/disease descriptions. FREE, no key needed.
"""

from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


def _make_tavily_tool() -> object:
    """
    Wrap TavilySearch so the LLM only sees a single `query` parameter.

    The raw TavilySearch schema exposes `topic`, `search_depth`, etc., which
    causes the model to pass unsupported values (e.g. topic="health").
    By wrapping it in a @tool with only `query`, we strip those options from
    the schema entirely — the underlying call always uses safe defaults.
    """
    _tavily = TavilySearch(max_results=3, topic="general")

    @tool
    def web_search(query: str) -> str:
        """Search the web for current medical information, symptoms, or treatments."""
        return _tavily.invoke({"query": query})

    return web_search


def get_symptom_tools() -> list:
    """
    Tools for the Symptom Analysis Agent.

    Why these three?
    - web_search: finds current symptom combinations, triage guides, clinical Q&A
    - PubMed: grounds the agent in peer-reviewed evidence for differential diagnosis
    - Wikipedia: gives accessible, structured overviews of conditions by name
      (useful when the agent has identified a candidate condition and wants a
       quick structured description to include in its response)
    """
    pubmed = PubmedQueryRun()

    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=1000  # keep context window usage reasonable
    )
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    return [_make_tavily_tool(), pubmed, wiki]


def get_research_tools() -> list:
    """
    Tools for the Research Agent.

    Why only two?
    The research agent handles focused, specific queries (e.g. "what are the
    current treatment guidelines for Type 2 diabetes?"). Wikipedia is less
    useful here — PubMed and Tavily together give authoritative, current results.
    Fewer tools = cleaner reasoning traces.
    """
    pubmed = PubmedQueryRun()

    return [_make_tavily_tool(), pubmed]
