"""Tool-calling ReAct agent — Step 3 of the agentic rollout.

The LLM dynamically decides which services to invoke for a given query instead
of running the fixed pipeline. Uses vLLM's OpenAI-compatible tool-calling API.

Requires:
- VLLM_ENDPOINT set
- The served model to support tool/function calling (e.g. Llama-3.1-Instruct
  started with `--enable-auto-tool-choice --tool-call-parser llama3_json`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from services.llm import (
    DEFAULT_MODEL, _get_endpoint, is_llm_available,
    generate_and_validate_narrative,
)
from services.narrative import generate_narrative
from services.search import search_tavily
from services.market_data import get_snapshots_for_query, get_credit_spreads
from services.sentiment import analyze_sentiment
from services.fear_greed import get_cnn_fear_greed, get_crypto_fear_greed


MAX_AGENT_ITERATIONS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — JSON-schema for the LLM + Python impl for execution
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search the web for recent news headlines about a query. Returns a list of titles and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "query_type": {
                        "type": "string",
                        "enum": ["oil", "neocloud", "crypto", "ai_robotics", "credit", "ticker", "macro"],
                        "description": "Category hint for the search; affects which source mix is used.",
                    },
                },
                "required": ["query", "query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prices",
            "description": "Fetch current price snapshots (price, 1d%, 5d%) for a query. Use this to get market data before writing a narrative.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "query_type": {
                        "type": "string",
                        "enum": ["oil", "neocloud", "crypto", "ai_robotics", "credit", "ticker", "macro"],
                    },
                },
                "required": ["query", "query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Run FinBERT sentiment analysis on a list of headline texts. Call after search_news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["texts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_credit_spreads",
            "description": "Get current credit spread readings (HYG/LQD/JNK vs TLT benchmark). Use for credit-regime queries.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fear_greed",
            "description": "Get CNN-style and Crypto Fear & Greed indices. Use for regime/sentiment queries.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Call this LAST with a brief rationale when you have enough evidence. The orchestrator will run the template + validation pipeline on the collected data to produce the final narrative.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: why the collected evidence is sufficient.",
                    },
                },
                "required": ["rationale"],
            },
        },
    },
]


@dataclass
class AgentTrace:
    tool: str
    args: dict[str, Any]
    summary: str  # short human-readable result preview


@dataclass
class AgentResult:
    narrative: str
    validation: Any
    trace: list[AgentTrace]
    collected: dict[str, Any]
    query_type: str
    iterations: int
    stop_reason: str  # "finalized" | "max_iters" | "no_more_tools" | "llm_error"
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations — each mutates `collected` and returns a short JSON payload
# ─────────────────────────────────────────────────────────────────────────────

def _tool_search_news(collected: dict, query: str, query_type: str) -> dict:
    results = search_tavily(query, query_type)
    collected["results"] = results
    collected.setdefault("query_type", query_type)
    return {
        "count": len(results),
        "titles": [r.title for r in results[:5]],
    }


def _tool_get_prices(collected: dict, query: str, query_type: str) -> dict:
    snaps = get_snapshots_for_query(query, query_type)
    collected["snapshots"] = snaps
    collected.setdefault("query_type", query_type)
    return {
        "count": len(snaps),
        "prices": [
            {
                "ticker": s.ticker,
                "price": s.price,
                "change_1d_pct": s.change_1d_pct,
                "change_5d_pct": s.change_5d_pct,
            }
            for s in snaps if not s.error
        ],
    }


def _tool_analyze_sentiment(collected: dict, texts: list[str]) -> dict:
    summary = analyze_sentiment(texts or [])
    collected["sentiment"] = summary
    return {
        "avg_score": summary.avg_score,
        "positive": summary.positive,
        "negative": summary.negative,
        "neutral": summary.neutral,
        "mode": summary.mode,
    }


def _tool_get_credit_spreads(collected: dict) -> dict:
    spreads = get_credit_spreads()
    collected["credit_spreads"] = spreads
    return {
        "spreads": [
            {"name": sp.name, "spread": sp.spread,
             "1d_change": sp.spread_1d_change,
             "interpretation": sp.interpretation}
            for sp in spreads
        ],
    }


def _tool_get_fear_greed(collected: dict) -> dict:
    cnn = get_cnn_fear_greed()
    crypto = get_crypto_fear_greed()
    collected["fear_greed"] = {"cnn": cnn, "crypto": crypto}
    return {
        "cnn": {"score": cnn.score, "classification": cnn.classification},
        "crypto": {"score": crypto.score, "classification": crypto.classification},
    }


def _tool_finalize(collected: dict, rationale: str) -> dict:
    collected["_finalize_rationale"] = rationale
    return {"acknowledged": True, "rationale": rationale}


TOOL_DISPATCH: dict[str, Callable[..., dict]] = {
    "search_news": _tool_search_news,
    "get_prices": _tool_get_prices,
    "analyze_sentiment": _tool_analyze_sentiment,
    "get_credit_spreads": _tool_get_credit_spreads,
    "get_fear_greed": _tool_get_fear_greed,
    "finalize": _tool_finalize,
}


# ─────────────────────────────────────────────────────────────────────────────
# The agent loop
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a macro market intelligence agent with access to tools.

Plan a minimal sequence of tool calls to gather evidence for the user's query, then call `finalize` to hand off to the narrative generator.

Guidelines:
- Usually call `get_prices` and `search_news` first.
- Call `analyze_sentiment` with the headline TITLES from search_news if sentiment matters.
- Call `get_credit_spreads` only for credit/risk-appetite queries.
- Call `get_fear_greed` only for broad regime/mood queries.
- Do NOT repeat a tool call with identical arguments.
- Call `finalize` as soon as you have enough; aim for 2-4 tool calls total.
"""


def _call_vllm_with_tools(messages: list[dict]) -> dict:
    endpoint = _get_endpoint()
    if not endpoint:
        raise RuntimeError("VLLM_ENDPOINT not configured")

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": os.getenv("VLLM_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 400,
    }
    r = requests.post(
        f"{endpoint}/chat/completions", headers=headers, json=payload, timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _truncate(obj: Any, limit: int = 80) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def run_agent(query: str, query_type_hint: str = "macro") -> AgentResult:
    """Run the tool-calling agent. Raises if vLLM is unavailable."""
    if not is_llm_available():
        raise RuntimeError(
            "Agent mode requires VLLM_ENDPOINT with tool-calling support "
            "(e.g. vllm serve ... --enable-auto-tool-choice --tool-call-parser llama3_json)."
        )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nHint (may ignore): query_type={query_type_hint}"},
    ]
    collected: dict[str, Any] = {"query_type": query_type_hint}
    trace: list[AgentTrace] = []
    stop_reason = "max_iters"
    rationale = ""

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        try:
            response = _call_vllm_with_tools(messages)
        except Exception as e:
            stop_reason = f"llm_error: {e}"
            break

        msg = response["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            stop_reason = "no_more_tools"
            break

        finalized = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_DISPATCH.get(name)
            if impl is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = impl(collected, **args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

            trace.append(AgentTrace(tool=name, args=args, summary=_truncate(result)))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": json.dumps(result, default=str)[:1500],
            })

            if name == "finalize":
                finalized = True
                rationale = args.get("rationale", "")

        if finalized:
            stop_reason = "finalized"
            break

    # Hand collected evidence to the existing narrative + validation pipeline
    narrative, validation = generate_and_validate_narrative(
        topic=query,
        query_type=collected.get("query_type", query_type_hint),
        results=collected.get("results", []),
        snapshots=collected.get("snapshots", []),
        sentiment=collected.get("sentiment") or analyze_sentiment([]),
        template_fallback_fn=generate_narrative,
    )

    return AgentResult(
        narrative=narrative,
        validation=validation,
        trace=trace,
        collected=collected,
        query_type=collected.get("query_type", query_type_hint),
        iterations=iteration,
        stop_reason=stop_reason,
        rationale=rationale,
    )
