"""Agent tool registry.

A single `Tool` dataclass holds the JSON-schema (for the LLM) and the Python
implementation (for execution). All tools live in this file; agent.py just
imports the registry and dispatches by name.

Each `impl(collected, **args)` mutates `collected` (the agent's evidence
dict) and returns a short dict that the LLM sees as the tool result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from services.search import search_tavily
from services.market_data import get_snapshots_for_query, get_credit_spreads
from services.sentiment import analyze_sentiment
from services.fear_greed import get_cnn_fear_greed, get_crypto_fear_greed
from services.fred import fetch_series as fred_fetch_series, fetch_default_panel as fred_panel
from services.cot import fetch_cot, fetch_default_panel as cot_panel
from services.options import get_iv_snapshot


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    impl: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def get(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in _REGISTRY.values()]


def names() -> list[str]:
    return list(_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def _tool_search_news(collected: dict, query: str, query_type: str) -> dict:
    results = search_tavily(query, query_type)
    collected["results"] = results
    collected.setdefault("query_type", query_type)
    return {"count": len(results), "titles": [r.title for r in results[:5]]}


def _tool_get_prices(collected: dict, query: str, query_type: str) -> dict:
    snaps = get_snapshots_for_query(query, query_type)
    collected["snapshots"] = snaps
    collected.setdefault("query_type", query_type)
    return {
        "count": len(snaps),
        "prices": [
            {"ticker": s.ticker, "price": s.price,
             "change_1d_pct": s.change_1d_pct, "change_5d_pct": s.change_5d_pct}
            for s in snaps if not s.error
        ],
    }


def _tool_analyze_sentiment(collected: dict, texts: list[str]) -> dict:
    summary = analyze_sentiment(texts or [])
    collected["sentiment"] = summary
    return {
        "avg_score": summary.avg_score, "positive": summary.positive,
        "negative": summary.negative, "neutral": summary.neutral, "mode": summary.mode,
    }


def _tool_get_credit_spreads(collected: dict) -> dict:
    spreads = get_credit_spreads()
    collected["credit_spreads"] = spreads
    return {"spreads": [{"name": sp.name, "spread": sp.spread,
                          "1d_change": sp.spread_1d_change,
                          "interpretation": sp.interpretation}
                         for sp in spreads]}


def _tool_get_fear_greed(collected: dict) -> dict:
    cnn = get_cnn_fear_greed()
    crypto = get_crypto_fear_greed()
    collected["fear_greed"] = {"cnn": cnn, "crypto": crypto}
    return {
        "cnn": {"score": cnn.score, "classification": cnn.classification},
        "crypto": {"score": crypto.score, "classification": crypto.classification},
    }


def _tool_get_macro_series(collected: dict, series_id: str) -> dict:
    s = fred_fetch_series(series_id)
    collected.setdefault("macro", {})[series_id] = s
    if s.error or not s.latest:
        return {"series_id": series_id, "error": s.error or "no data"}
    date, val = s.latest
    return {
        "series_id": series_id, "name": s.name, "units": s.units,
        "latest_date": date, "latest_value": val,
        "change_5d": s.change(5), "change_20d": s.change(20),
    }


def _tool_get_macro_panel(collected: dict) -> dict:
    panel = fred_panel()
    collected["macro_panel"] = panel
    out = []
    for s in panel:
        if s.error or not s.latest:
            out.append({"series_id": s.series_id, "error": s.error or "no data"})
            continue
        date, val = s.latest
        out.append({
            "series_id": s.series_id, "name": s.name, "units": s.units,
            "latest_date": date, "latest_value": val,
            "change_5d": s.change(5), "change_20d": s.change(20),
        })
    return {"panel": out}


def _tool_get_positioning(collected: dict, market_code: str) -> dict:
    s = fetch_cot(market_code)
    collected.setdefault("positioning", {})[market_code] = s
    if s.error or s.net_noncomm is None:
        return {"market_code": market_code, "error": s.error or "no data"}
    return {
        "market_code": s.code, "name": s.name, "report_date": s.report_date,
        "net_spec": s.net_noncomm, "week_change": s.week_change,
        "z_score_52w": s.z_score,
        "crowded": (s.z_score is not None and abs(s.z_score) >= 2.0),
    }


def _tool_get_positioning_panel(collected: dict) -> dict:
    panel = cot_panel()
    collected["positioning_panel"] = panel
    out = []
    for s in panel:
        if s.error or s.net_noncomm is None:
            out.append({"market_code": s.code, "error": s.error or "no data"})
            continue
        out.append({
            "market_code": s.code, "name": s.name, "report_date": s.report_date,
            "net_spec": s.net_noncomm, "week_change": s.week_change,
            "z_score_52w": s.z_score,
            "crowded": (s.z_score is not None and abs(s.z_score) >= 2.0),
        })
    return {"panel": out}


def _tool_get_options_iv(collected: dict, ticker: str) -> dict:
    s = get_iv_snapshot(ticker)
    collected.setdefault("options_iv", {})[ticker.upper()] = s
    if s.error:
        return {"ticker": s.ticker, "error": s.error}
    return {
        "ticker": s.ticker, "spot": s.spot,
        "expiry": s.expiry, "days_to_expiry": s.days_to_expiry,
        "atm_strike": s.atm_strike,
        "call_iv": s.call_iv, "put_iv": s.put_iv, "avg_iv": s.avg_iv,
    }


def _tool_finalize(collected: dict, rationale: str) -> dict:
    collected["_finalize_rationale"] = rationale
    return {"acknowledged": True, "rationale": rationale}


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_TYPE_ENUM = ["oil", "neocloud", "crypto", "ai_robotics", "credit", "ticker", "macro"]

register(Tool(
    name="search_news",
    description="Search the web for recent news headlines about a query. Returns titles and snippets. Always call before analyze_sentiment.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "query_type": {"type": "string", "enum": _QUERY_TYPE_ENUM,
                            "description": "Category hint affecting the source mix."},
        },
        "required": ["query", "query_type"],
    },
    impl=_tool_search_news,
))

register(Tool(
    name="get_prices",
    description="Fetch current price snapshots (price, 1d%, 5d%) for a query. Use before writing the narrative.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "query_type": {"type": "string", "enum": _QUERY_TYPE_ENUM},
        },
        "required": ["query", "query_type"],
    },
    impl=_tool_get_prices,
))

register(Tool(
    name="analyze_sentiment",
    description="Run FinBERT sentiment on a list of headline texts. Call after search_news with the returned titles.",
    parameters={
        "type": "object",
        "properties": {
            "texts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["texts"],
    },
    impl=_tool_analyze_sentiment,
))

register(Tool(
    name="get_credit_spreads",
    description="Get current credit spread readings (HYG/LQD/JNK vs TLT). Use only for credit/risk-appetite queries.",
    parameters={"type": "object", "properties": {}},
    impl=_tool_get_credit_spreads,
))

register(Tool(
    name="get_fear_greed",
    description="Get CNN-style and Crypto Fear & Greed indices. Use for broad regime/mood queries.",
    parameters={"type": "object", "properties": {}},
    impl=_tool_get_fear_greed,
))

register(Tool(
    name="get_macro_series",
    description=(
        "Fetch a single FRED macro series with latest value and 5d/20d change. "
        "Useful when only one rate or index is relevant. "
        "Common ids: DGS10 (10Y yield), DGS2 (2Y), T10Y2Y (curve), DFF (fed funds), "
        "DTWEXBGS (broad USD), VIXCLS (VIX), BAMLH0A0HYM2 (HY OAS), T10YIE (breakeven), "
        "UNRATE, CPIAUCSL, DCOILWTICO."
    ),
    parameters={
        "type": "object",
        "properties": {
            "series_id": {"type": "string",
                           "description": "FRED series id, e.g. DGS10."},
        },
        "required": ["series_id"],
    },
    impl=_tool_get_macro_series,
))

register(Tool(
    name="get_macro_panel",
    description="Fetch the full FRED macro panel (rates, dollar, vol, credit, breakevens) in one call. Prefer this for any macro-flavored query.",
    parameters={"type": "object", "properties": {}},
    impl=_tool_get_macro_panel,
))

register(Tool(
    name="get_positioning",
    description=(
        "Fetch CFTC COT speculator positioning for a single market with 52w z-score. "
        "Codes: CL (WTI crude), GC (gold), HG (copper), ES (E-mini S&P 500), "
        "NQ (Nasdaq 100), 6E (Euro FX), VX (VIX). "
        "|z|≥2 = crowded extreme."
    ),
    parameters={
        "type": "object",
        "properties": {
            "market_code": {"type": "string"},
        },
        "required": ["market_code"],
    },
    impl=_tool_get_positioning,
))

register(Tool(
    name="get_positioning_panel",
    description="Fetch the full CFTC COT positioning panel across all tracked markets. Use to scan for crowded extremes.",
    parameters={"type": "object", "properties": {}},
    impl=_tool_get_positioning_panel,
))

register(Tool(
    name="get_options_iv",
    description=(
        "Fetch implied volatility for a ticker's nearest monthly expiry "
        "(≥21 days out). Returns ATM call IV, put IV, and the average "
        "(decimal, e.g. 0.18 = 18% annualized). Use to gauge expected vol; "
        "compare to realized vol or VIX for richness."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Equity/ETF symbol (e.g. SPY, NVDA, TLT)."},
        },
        "required": ["ticker"],
    },
    impl=_tool_get_options_iv,
))

register(Tool(
    name="finalize",
    description="Call LAST with a brief rationale once you have enough evidence. Hands off to the narrative + validation pipeline.",
    parameters={
        "type": "object",
        "properties": {
            "rationale": {"type": "string",
                           "description": "One sentence: why the collected evidence is sufficient."},
        },
        "required": ["rationale"],
    },
    impl=_tool_finalize,
))
