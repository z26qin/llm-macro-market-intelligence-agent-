"""Linear orchestrator — fallback path used when the agent is unavailable."""

from __future__ import annotations

from services.search import search_tavily, SearchResult
from services.market_data import (
    get_snapshots_for_query, PriceSnapshot,
    get_credit_spreads, CreditSpread,
)
from services.sentiment import analyze_sentiment, SentimentSummary
from services.narrative import generate_narrative
from services.llm import generate_and_validate_narrative


def run_analysis(query: str, query_type: str) -> dict:
    """Linear pipeline: search → market data → sentiment → narrative."""
    query = query.strip()
    if not query:
        return {"error": "Please enter a query."}

    results: list[SearchResult] = search_tavily(query, query_type)
    snapshots: list[PriceSnapshot] = get_snapshots_for_query(query, query_type)

    texts: list[str] = []
    for r in results:
        if r.title:
            texts.append(r.title)
        if r.snippet:
            texts.append(r.snippet[:300])
    sentiment: SentimentSummary = analyze_sentiment(texts)

    credit_spreads: list[CreditSpread] = []
    if query_type == "credit":
        credit_spreads = get_credit_spreads()

    narrative, validation = generate_and_validate_narrative(
        topic=query, query_type=query_type,
        results=results, snapshots=snapshots, sentiment=sentiment,
        template_fallback_fn=generate_narrative,
    )

    return {
        "results": results,
        "snapshots": snapshots,
        "sentiment": sentiment,
        "narrative": narrative,
        "validation": validation,
        "credit_spreads": credit_spreads,
        "query_type": query_type,
    }
