"""Nebius Token Factory LLM integration for narrative generation."""

from __future__ import annotations

import os
import json
import requests

from services.market_data import PriceSnapshot
from services.search import SearchResult
from services.sentiment import SentimentSummary


NEBIUS_API_URL = "https://api.studio.nebius.com/v1/chat/completions"


def _format_price_data(snapshots: list[PriceSnapshot]) -> str:
    """Format price snapshots for the prompt."""
    lines = []
    for s in snapshots:
        if s.error:
            lines.append(f"- {s.ticker}: data unavailable ({s.error})")
        else:
            d1 = f"{s.change_1d_pct:+.2f}%" if s.change_1d_pct is not None else "n/a"
            d5 = f"{s.change_5d_pct:+.2f}%" if s.change_5d_pct is not None else "n/a"
            lines.append(f"- {s.name} ({s.ticker}): ${s.price} | 1d: {d1} | 5d: {d5}")
    return "\n".join(lines) if lines else "No market data available."


def _format_headlines(results: list[SearchResult]) -> str:
    """Format search results as headlines for the prompt."""
    lines = []
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. {r.title}")
        if r.snippet:
            lines.append(f"   {r.snippet[:150]}...")
    return "\n".join(lines) if lines else "No headlines available."


def _format_sentiment(sentiment: SentimentSummary) -> str:
    """Format sentiment summary for the prompt."""
    return (
        f"Average score: {sentiment.avg_score:+.3f}\n"
        f"Positive: {sentiment.positive}, Negative: {sentiment.negative}, "
        f"Neutral: {sentiment.neutral}\n"
        f"Mode: {sentiment.mode}"
    )


def _build_prompt(
    topic: str,
    price_data: str,
    headlines: str,
    sentiment_summary: str,
) -> str:
    """Build the prompt for the LLM."""
    return f"""You are a macro market analyst.

Explain the likely drivers behind today's move in {topic}.

Market data:
{price_data}

News evidence:
{headlines}

Sentiment summary:
{sentiment_summary}

Produce a short explanation with the following sections:
1. Move Summary
2. Likely Drivers
3. Market Interpretation
4. Confidence Level

Be concise and analytical."""


def is_nebius_available() -> bool:
    """Check if Nebius API key is configured."""
    return bool(os.getenv("NEBIUS_API_KEY"))


def generate_market_narrative(
    topic: str,
    price_data: dict,
    headlines: list,
    sentiment_summary: dict,
) -> str:
    """Generate a market narrative using Nebius Token Factory.

    Args:
        topic: The market topic being analyzed (e.g., "NVDA", "oil")
        price_data: Dict containing price snapshots (list of PriceSnapshot objects)
        headlines: List of SearchResult objects with news headlines
        sentiment_summary: SentimentSummary object with sentiment analysis

    Returns:
        Generated narrative string from the LLM
    """
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY environment variable is not set")

    # Format inputs for the prompt
    formatted_prices = _format_price_data(price_data.get("snapshots", []))
    formatted_headlines = _format_headlines(headlines)
    formatted_sentiment = _format_sentiment(sentiment_summary)

    prompt = _build_prompt(
        topic=topic,
        price_data=formatted_prices,
        headlines=formatted_headlines,
        sentiment_summary=formatted_sentiment,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            NEBIUS_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Nebius API request timed out")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Nebius API request failed: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Nebius API response format: {e}")


def generate_narrative_with_fallback(
    topic: str,
    query_type: str,
    results: list[SearchResult],
    snapshots: list[PriceSnapshot],
    sentiment: SentimentSummary,
    template_fallback_fn,
) -> str:
    """Generate narrative using Nebius LLM, with fallback to template generator.

    Args:
        topic: The query/topic being analyzed
        query_type: Type of query (oil, neocloud, ticker, macro)
        results: Search results from Tavily
        snapshots: Price snapshots from yfinance
        sentiment: Sentiment analysis summary
        template_fallback_fn: Fallback function to use if Nebius is unavailable

    Returns:
        Generated narrative string
    """
    if not is_nebius_available():
        return template_fallback_fn(topic, query_type, results, snapshots, sentiment)

    try:
        return generate_market_narrative(
            topic=topic,
            price_data={"snapshots": snapshots},
            headlines=results,
            sentiment_summary=sentiment,
        )
    except Exception as e:
        # Log the error and fall back to template
        print(f"[llm_nebius] Nebius API error: {e}. Falling back to template.")
        return template_fallback_fn(topic, query_type, results, snapshots, sentiment)
