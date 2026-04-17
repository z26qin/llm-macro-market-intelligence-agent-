"""Deterministic narrative generator — template + evidence synthesis.

Produces concise, hedge-fund-style market commentary without an external LLM.
"""

from __future__ import annotations

from services.search import SearchResult
from services.market_data import PriceSnapshot
from services.sentiment import SentimentSummary


def _direction_word(pct: float | None) -> str:
    if pct is None:
        return "flat (no data)"
    if pct > 1.5:
        return "sharply higher"
    if pct > 0.3:
        return "modestly higher"
    if pct < -1.5:
        return "sharply lower"
    if pct < -0.3:
        return "modestly lower"
    return "roughly flat"


def _sentiment_read(summary: SentimentSummary) -> str:
    if summary.avg_score > 0.25:
        return "decidedly bullish"
    if summary.avg_score > 0.05:
        return "leaning positive"
    if summary.avg_score < -0.25:
        return "decidedly bearish"
    if summary.avg_score < -0.05:
        return "leaning negative"
    return "mixed / neutral"


def _move_classification(snapshots: list[PriceSnapshot]) -> str:
    """Heuristic: if most tickers move the same way, it's macro/sector-wide."""
    if not snapshots:
        return "indeterminate"
    directions = []
    for s in snapshots:
        if s.change_1d_pct is not None:
            directions.append(s.change_1d_pct)
    if not directions:
        return "indeterminate"
    pos = sum(1 for d in directions if d > 0.3)
    neg = sum(1 for d in directions if d < -0.3)
    total = len(directions)
    if pos >= total * 0.7:
        return "broad-based rally — likely macro or sector-driven"
    if neg >= total * 0.7:
        return "broad-based sell-off — likely macro or sector-driven"
    return "divergent moves — likely idiosyncratic or stock-specific"


def _top_drivers(results: list[SearchResult], n: int = 3) -> list[tuple[int, str]]:
    """Extract top headline drivers with source numbers for citation."""
    drivers = []
    for i, r in enumerate(results[:n], 1):
        title = r.title.strip()
        if title:
            drivers.append((i, title))
    return drivers


def generate_narrative(
    query: str,
    query_type: str,
    results: list[SearchResult],
    snapshots: list[PriceSnapshot],
    sentiment: SentimentSummary,
) -> str:
    """Build a concise market narrative from all evidence."""
    lines: list[str] = []

    # ── Move summary ─────────────────────────────────────────────────────
    lines.append("## Move Summary")
    for s in snapshots:
        if s.error:
            lines.append(f"- **{s.ticker}**: data unavailable ({s.error})")
        else:
            d1 = f"{s.change_1d_pct:+.2f}%" if s.change_1d_pct is not None else "n/a"
            d5 = f"{s.change_5d_pct:+.2f}%" if s.change_5d_pct is not None else "n/a"
            lines.append(
                f"- **{s.name}** ({s.ticker}): ${s.price}  |  1d: {d1}  |  5d: {d5}  "
                f"({_direction_word(s.change_1d_pct)})"
            )

    # ── Key drivers ──────────────────────────────────────────────────────
    drivers = _top_drivers(results)
    lines.append("\n## Key Drivers")
    if drivers:
        for source_num, headline in drivers:
            lines.append(f"{source_num}. {headline} [Source {source_num}]")
    else:
        lines.append("- No headline drivers retrieved.")

    # ── Sentiment read ───────────────────────────────────────────────────
    lines.append("\n## Sentiment Read")
    read = _sentiment_read(sentiment)

    # Reference the sources that contributed to sentiment
    source_refs = " ".join([f"[Source {i}]" for i, _ in drivers]) if drivers else ""
    lines.append(
        f"Headline sentiment is **{read}** based on analysis of retrieved news {source_refs}. "
        f"Aggregate score: {sentiment.avg_score:+.3f} "
        f"(+{sentiment.positive} positive / -{sentiment.negative} negative / ~{sentiment.neutral} neutral, "
        f"mode: {sentiment.mode})."
    )

    # ── Market interpretation ────────────────────────────────────────────
    classification = _move_classification(snapshots)
    lines.append("\n## Market Interpretation")
    lines.append(f"Price action suggests **{classification}**.")

    # Add source references to divergence detection
    source_refs_str = ", ".join([f"[Source {i}]" for i, _ in drivers]) if drivers else ""

    if sentiment.avg_score > 0.05 and any(
        (s.change_1d_pct or 0) < -0.3 for s in snapshots
    ):
        lines.append(
            f"Note: positive sentiment from headlines {source_refs_str} diverges from negative price action — "
            "watch for potential mean reversion or delayed headline impact."
        )
    elif sentiment.avg_score < -0.05 and any(
        (s.change_1d_pct or 0) > 0.3 for s in snapshots
    ):
        lines.append(
            f"Note: negative sentiment from headlines {source_refs_str} diverges from positive price action — "
            "market may be looking through near-term headwinds."
        )

    # ── Confidence Assessment ────────────────────────────────────────────
    lines.append("\n## Confidence Assessment")

    # Determine confidence level based on data quality
    confidence_factors = []
    confidence_level = "HIGH"

    # Check data availability
    available_prices = sum(1 for s in snapshots if not s.error)
    if available_prices < len(snapshots) * 0.5:
        confidence_level = "MEDIUM"
        confidence_factors.append("limited price data availability")

    # Check headline coverage
    if len(results) < 3:
        confidence_level = "MEDIUM" if confidence_level == "HIGH" else "LOW"
        confidence_factors.append("limited headline coverage")

    # Check sentiment mode
    if sentiment.mode == "mock":
        confidence_level = "MEDIUM" if confidence_level == "HIGH" else "LOW"
        confidence_factors.append("using keyword-based sentiment (not FinBERT)")

    lines.append(f"Overall confidence: **{confidence_level}**")
    if confidence_factors:
        lines.append(f"Limiting factors: {', '.join(confidence_factors)}")

    # ── Caveats ──────────────────────────────────────────────────────────
    lines.append("\n## Caveats")
    caveats = [
        "Prototype analysis — not investment advice.",
        f"Based on {len(results)} retrieved headlines {source_refs_str}; coverage may be incomplete.",
    ]
    if sentiment.mode == "mock":
        caveats.append("Sentiment scored via keyword heuristic (mock mode).")
    for c in caveats:
        lines.append(f"- {c}")

    return "\n".join(lines)
