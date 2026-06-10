"""
Web search tool for the text agent.

CRITICAL DESIGN POINT (contamination control):
The NEJM Image Challenge is public. A naive search on the answer-option strings
can retrieve pages that state the correct answer, turning "clinical reasoning"
into "answer-key lookup". To keep the experiment interpretable we:

  1. Log EVERY query and EVERY returned snippet into the graph state, so a
     post-hoc audit can measure how often a retrieval leaked the answer.
  2. Make search toggleable (use_web_search flag) so the *paired control*
     (identical graph, search OFF) can be run and compared. The with-search
     number is only meaningful relative to that control.
  3. Bias queries toward disease/finding knowledge rather than the verbatim
     case text (see node prompt), reducing direct hits on quiz pages.

Provider: Tavily (langchain-tavily). To run keyless, swap _raw_search for a
DuckDuckGo implementation — this file is the only thing that changes.
"""

import os
from typing import Tuple, List


def _raw_search(query: str, max_results: int = 3) -> List[dict]:
    """Return a list of {title, content, url} dicts. Provider-specific."""
    from langchain_tavily import TavilySearch
    tool = TavilySearch(max_results=max_results, api_key=os.getenv("TAVILY_API_KEY"))
    raw = tool.invoke({"query": query})
    # TavilySearch returns a dict with a "results" list
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [
        {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
        for r in results
    ]


def run_search(query: str, max_results: int = 3) -> Tuple[str, List[dict]]:
    """
    Execute one search. Returns (formatted_text_for_model, raw_snippets_for_audit).

    Never raises into the graph — on failure returns an empty result and an
    error marker in the audit list, so one flaky search can't kill a 6-hour run.
    """
    try:
        snippets = _raw_search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 — deliberately broad; logging not crashing
        return f"[search failed: {e}]", [{"error": str(e), "query": query}]

    if not snippets:
        return "[no results]", []

    formatted = "\n\n".join(
        f"- {s['title']}: {s['content'][:400]}" for s in snippets
    )
    return formatted, snippets