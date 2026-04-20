"""LLM integration for narrative generation via vLLM (OpenAI-compatible API).

Supports self-hosted vLLM on:
- Nebius Cloud VMs
- AWS EC2 (p4d, p5 instances)
- Any GPU infrastructure

Environment variables:
- VLLM_ENDPOINT: vLLM server URL (e.g., http://localhost:8000/v1)
- VLLM_API_KEY: Optional API key if server requires auth
- VLLM_MODEL: Model name (default: meta-llama/Llama-3.3-70B-Instruct)
"""

from __future__ import annotations

import os
import requests

from services.market_data import PriceSnapshot
from services.search import SearchResult
from services.sentiment import SentimentSummary  # noqa: F401 — used in type hints below
from services.validation import validate_narrative, ValidationResult


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Self-correction loop config
MAX_CORRECTION_ATTEMPTS = 3     # initial + up to 2 corrections
MIN_ACCEPTABLE_CONFIDENCE = 60.0


def _get_endpoint() -> str:
    """Get the vLLM endpoint URL."""
    endpoint = os.getenv("VLLM_ENDPOINT", "").rstrip("/")
    if endpoint and not endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/v1"
    return endpoint


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
    """Format search results as numbered headlines for the prompt (enables citation)."""
    lines = []
    for i, r in enumerate(results[:5], 1):
        lines.append(f"[Source {i}] {r.title}")
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
    """Build the prompt for the LLM with citation enforcement and uncertainty quantification."""
    return f"""You are a macro market analyst. Your task is to explain today's market moves using ONLY the provided evidence.

CRITICAL REQUIREMENTS:
1. CITE EVERY CLAIM: For every factual claim you make, you MUST cite the specific source using [Source N] notation
2. USE EXACT NUMBERS: When mentioning price changes, use the exact percentages from the market data
3. QUANTIFY CONFIDENCE: Rate your confidence for each section (HIGH/MEDIUM/LOW)
4. NO SPECULATION: If evidence is insufficient, state "INSUFFICIENT DATA" instead of guessing

Available evidence:

Market data (use exact percentages):
{price_data}

News sources (cite as [Source N]):
{headlines}

Sentiment summary:
{sentiment_summary}

Generate a structured analysis with these sections:

1. MOVE SUMMARY (Confidence: HIGH/MEDIUM/LOW)
   - State the price movements using exact percentages from market data
   - Cite any news sources that explain the moves [Source N]

2. LIKELY DRIVERS (Confidence: HIGH/MEDIUM/LOW)
   - List 2-3 key drivers, each with a citation [Source N]
   - If multiple sources support a driver, cite all relevant sources

3. MARKET INTERPRETATION (Confidence: HIGH/MEDIUM/LOW)
   - Provide your interpretation of market sentiment
   - Support with citations to news sources [Source N]

4. OVERALL CONFIDENCE SCORE: [0-100]
   - Rate your overall confidence in this analysis (0-100)
   - Explain what factors limit your confidence

REMEMBER:
- Every claim needs a [Source N] citation
- Use exact numbers from market data
- State "INSUFFICIENT DATA" if unsupported
- Be concise (200-300 words total)"""


def is_llm_available() -> bool:
    """Check if vLLM endpoint is configured."""
    return bool(_get_endpoint())


def generate_market_narrative(
    topic: str,
    price_data: dict,
    headlines: list[SearchResult],
    sentiment_summary: SentimentSummary,
) -> str:
    """Generate a market narrative using vLLM.

    Args:
        topic: The market topic being analyzed (e.g., "NVDA", "oil")
        price_data: Dict with key "snapshots" containing list of PriceSnapshot objects
        headlines: List of SearchResult objects with news headlines
        sentiment_summary: SentimentSummary object with sentiment analysis

    Returns:
        Generated narrative string from the LLM
    """
    endpoint = _get_endpoint()
    if not endpoint:
        raise ValueError("VLLM_ENDPOINT environment variable is not set")

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

    headers = {"Content-Type": "application/json"}

    # Add auth header if API key is set
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model = os.getenv("VLLM_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            f"{endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("vLLM request timed out")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"vLLM request failed: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected vLLM response format: {e}")


def _build_correction_prompt(
    topic: str,
    price_data: str,
    headlines: str,
    sentiment_summary: str,
    previous_narrative: str,
    validation: ValidationResult,
) -> str:
    """Build a correction prompt that cites specific validator failures."""
    errors_block = "\n".join(f"- {e}" for e in validation.errors) or "- (none)"
    warnings_block = "\n".join(f"- {w}" for w in validation.warnings) or "- (none)"
    return f"""You are a macro market analyst. Your previous analysis FAILED validation and must be rewritten.

PREVIOUS ANALYSIS (reject and rewrite, do not merely edit):
---
{previous_narrative}
---

VALIDATOR ERRORS (must fix):
{errors_block}

VALIDATOR WARNINGS (should fix):
{warnings_block}

PREVIOUS CONFIDENCE SCORE: {validation.confidence_score:.0f}/100 (minimum acceptable: {MIN_ACCEPTABLE_CONFIDENCE:.0f})

Common causes of failure:
- Numerical mismatches: invented a percentage not present in the market data.
  → Fix: use ONLY the exact numbers from the market data block below.
- Missing citations: factual claims without [Source N] tags.
  → Fix: every factual sentence must end with at least one [Source N].
- Invalid citations: cited [Source N] where N exceeds available sources.
  → Fix: only cite sources that exist below.

Available evidence (same as before — rewrite strictly from this):

Market data (use these exact percentages, no others):
{price_data}

News sources (cite as [Source N], N must be valid):
{headlines}

Sentiment summary:
{sentiment_summary}

Rewrite the analysis using the same 4-section structure:
1. MOVE SUMMARY (Confidence: HIGH/MEDIUM/LOW)
2. LIKELY DRIVERS (Confidence: HIGH/MEDIUM/LOW)
3. MARKET INTERPRETATION (Confidence: HIGH/MEDIUM/LOW)
4. OVERALL CONFIDENCE SCORE: [0-100]

REMEMBER:
- Every factual claim needs [Source N]
- Use ONLY exact numbers from the market data
- State "INSUFFICIENT DATA" if evidence is missing
- Be concise (200-300 words)"""


def _call_vllm(prompt: str) -> str:
    """Call vLLM /chat/completions with a single user prompt, return text."""
    endpoint = _get_endpoint()
    if not endpoint:
        raise ValueError("VLLM_ENDPOINT environment variable is not set")

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": os.getenv("VLLM_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    response = requests.post(
        f"{endpoint}/chat/completions", headers=headers, json=payload, timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _snapshots_to_market_data(snapshots: list[PriceSnapshot]) -> list[dict]:
    return [{
        "ticker": s.ticker,
        "current_price": s.price if not s.error else None,
        "change_1d": s.change_1d_pct,
        "change_5d": s.change_5d_pct,
    } for s in snapshots]


def generate_and_validate_narrative(
    topic: str,
    query_type: str,
    results: list[SearchResult],
    snapshots: list[PriceSnapshot],
    sentiment: SentimentSummary,
    template_fallback_fn,
) -> tuple[str, ValidationResult]:
    """Generate narrative and validate it, with self-correcting retry loop.

    When vLLM is available and validation fails (or confidence < 60), the LLM
    is re-prompted with the validator's structured errors/warnings. Caps at
    MAX_CORRECTION_ATTEMPTS total attempts. Template fallback path does not
    retry (no LLM available to learn from the feedback).

    Returns the best narrative + ValidationResult. `validation.attempts`
    reports how many LLM calls were made.
    """
    market_data = _snapshots_to_market_data(snapshots)
    num_sources = min(len(results), 5)

    # Attempt 1 — uses fallback path (template if LLM unavailable)
    narrative = generate_narrative_with_fallback(
        topic, query_type, results, snapshots, sentiment, template_fallback_fn
    )
    validation = validate_narrative(
        narrative=narrative, market_data=market_data,
        num_sources=num_sources, sentiment_score=sentiment.avg_score,
    )
    validation.attempts = 1

    # No retries if LLM isn't available (template produces identical output)
    if not is_llm_available():
        return narrative, validation

    # Pre-format static inputs once for correction prompts
    price_block = _format_price_data(snapshots)
    headlines_block = _format_headlines(results)
    sentiment_block = _format_sentiment(sentiment)

    # Track best result across attempts so a worse retry can't overwrite a better original
    best_narrative, best_validation = narrative, validation

    attempt = 1
    while (
        (not validation.passed or validation.confidence_score < MIN_ACCEPTABLE_CONFIDENCE)
        and attempt < MAX_CORRECTION_ATTEMPTS
    ):
        attempt += 1
        correction_prompt = _build_correction_prompt(
            topic, price_block, headlines_block, sentiment_block,
            previous_narrative=narrative, validation=validation,
        )
        try:
            narrative = _call_vllm(correction_prompt)
        except Exception as e:
            print(f"[llm] correction attempt {attempt} failed: {e}")
            break

        validation = validate_narrative(
            narrative=narrative, market_data=market_data,
            num_sources=num_sources, sentiment_score=sentiment.avg_score,
        )
        validation.attempts = attempt

        if validation.confidence_score > best_validation.confidence_score:
            best_narrative, best_validation = narrative, validation

    # Return the best version seen, not just the last
    best_validation.attempts = attempt
    return best_narrative, best_validation


def generate_narrative_with_fallback(
    topic: str,
    query_type: str,
    results: list[SearchResult],
    snapshots: list[PriceSnapshot],
    sentiment: SentimentSummary,
    template_fallback_fn,
) -> str:
    """Generate narrative using vLLM, with fallback to template generator.

    Args:
        topic: The query/topic being analyzed
        query_type: Type of query (oil, neocloud, ticker, macro)
        results: Search results from Tavily
        snapshots: Price snapshots from yfinance
        sentiment: Sentiment analysis summary
        template_fallback_fn: Fallback function to use if vLLM is unavailable

    Returns:
        Generated narrative string
    """
    if not is_llm_available():
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
        print(f"[llm] vLLM error: {e}. Falling back to template.")
        return template_fallback_fn(topic, query_type, results, snapshots, sentiment)
