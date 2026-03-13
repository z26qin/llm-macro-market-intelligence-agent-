"""Tavily web search integration for retrieving market-relevant headlines."""

from __future__ import annotations

from dataclasses import dataclass
from utils.config import TAVILY_API_KEY, SEARCH_MAX_RESULTS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published_date: str | None = None  # ISO format date string


def _build_search_query(query: str, query_type: str) -> str:
    """Build a search-engine-friendly query string."""
    templates = {
        "oil": f"why is oil price moving today {query} crude energy markets",
        "neocloud": f"{query} neocloud AI infrastructure stock market news today",
        "ticker": f"why is {query} stock moving today market news",
        "macro": f"{query} macro market narrative today economy",
    }
    return templates.get(query_type, f"{query} market news today")


def search_tavily(query: str, query_type: str) -> list[SearchResult]:
    """Run a Tavily search and return structured results.

    Falls back to empty results if API key is missing or call fails.
    """
    search_query = _build_search_query(query, query_type)

    if not TAVILY_API_KEY:
        return _mock_results(search_query)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=search_query,
            max_results=SEARCH_MAX_RESULTS,
            search_depth="advanced",
            include_answer=False,
        )
        results: list[SearchResult] = []
        for r in response.get("results", []):
            # Try both snake_case and camelCase for published date
            pub_date = r.get("published_date") or r.get("publishedDate")
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")[:500],
                    published_date=pub_date,
                )
            )
        # Sort by published_date (most recent first), None dates go last
        results.sort(key=lambda x: x.published_date or "", reverse=True)
        return results
    except Exception as e:
        print(f"[search] Tavily error: {e}")
        return _mock_results(search_query)


def _mock_results(query: str) -> list[SearchResult]:
    """Return placeholder results when Tavily is unavailable."""
    return [
        SearchResult(
            title=f"[Mock] No live search — query was: {query[:80]}",
            url="",
            snippet="Tavily API key not set or search failed. Set TAVILY_API_KEY to enable live search.",
        )
    ]
