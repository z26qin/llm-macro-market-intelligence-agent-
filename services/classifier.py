"""Query classifier agent.

Uses vLLM to classify a free-text query into one of the supported query types
and extract any inline tickers. Falls back to a keyword heuristic when vLLM
is unavailable or the LLM response is unparseable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import requests

from services.llm import DEFAULT_MODEL, _get_endpoint, is_llm_available


VALID_QUERY_TYPES = {
    "oil", "neocloud", "crypto", "ai_robotics", "credit", "ticker", "macro",
}

# Keyword hints for the fallback heuristic (checked in order; first match wins).
_KEYWORD_HINTS: list[tuple[str, str]] = [
    (r"\b(oil|crude|wti|brent|opec|cl=f|bz=f)\b", "oil"),
    (r"\b(credit|spread|hyg|lqd|jnk|bkln|emb|junk)\b", "credit"),
    (r"\b(btc|bitcoin|crypto|mstr|mstu|bmnr|ibit)\b", "crypto"),
    (r"\b(tsla|tesla|robot|robotics|autonomous)\b", "ai_robotics"),
    (r"\b(nvda|nbis|crwv|orcl|msft|cifr|mrvl|neocloud|ai infra|gpu|hyperscaler)\b", "neocloud"),
    (r"\b(fed|fomc|cpi|tariff|yield|dxy|spy|qqq|macro|recession|inflation)\b", "macro"),
]

# Obvious ticker patterns (uppercase 1-5 letters, optional =F for futures)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:=F|-USD)?)\b")


@dataclass
class ClassificationResult:
    query_type: str
    tickers: list[str]
    reasoning: str
    used_llm: bool


def _heuristic_classify(query: str) -> ClassificationResult:
    q_lower = query.lower()
    tickers = _extract_tickers(query)

    for pattern, label in _KEYWORD_HINTS:
        if re.search(pattern, q_lower):
            return ClassificationResult(
                query_type=label,
                tickers=tickers,
                reasoning=f"Keyword match: '{pattern}' → {label}",
                used_llm=False,
            )

    # No keyword hit — if we saw a ticker-like token, treat as ticker query
    if tickers:
        return ClassificationResult(
            query_type="ticker",
            tickers=tickers,
            reasoning=f"No theme keywords matched; treating uppercase token(s) {tickers} as ticker(s)",
            used_llm=False,
        )

    return ClassificationResult(
        query_type="macro",
        tickers=[],
        reasoning="No keywords or tickers matched; defaulting to macro.",
        used_llm=False,
    )


def _extract_tickers(query: str) -> list[str]:
    # Skip common English words that happen to be uppercase-ish (unlikely in queries but cheap)
    noise = {"THE", "AND", "FOR", "VS", "VS.", "A", "I"}
    return [t for t in _TICKER_RE.findall(query) if t not in noise]


def _build_classifier_prompt(query: str) -> str:
    return f"""You are a query classifier for a market intelligence tool. Classify the user's query into EXACTLY ONE of these types:

- oil: crude oil, energy, OPEC, WTI/Brent
- neocloud: NVDA, NBIS, CRWV, ORCL, MSFT, CIFR, MRVL, AI infra, GPU cloud, hyperscalers
- crypto: Bitcoin, crypto miners (MSTR, BMNR), IBIT, crypto market
- ai_robotics: TSLA, robotics, autonomous systems
- credit: HY/IG spreads, HYG, LQD, JNK, BKLN, EMB, credit risk
- macro: Fed, CPI, tariffs, DXY, recession, broad index (SPY/QQQ) regime
- ticker: any other specific stock ticker that doesn't fit above

User query: "{query}"

Respond with ONLY a JSON object (no prose, no markdown fences) in this exact schema:
{{"query_type": "<one of the 7 types>", "tickers": ["<TICKER1>", ...], "reasoning": "<one short sentence>"}}

The "tickers" array should include any specific ticker symbols mentioned in the query (uppercase, e.g. "NVDA", "BTC-USD", "CL=F"). Empty array if none."""


def _call_classifier_llm(query: str) -> dict | None:
    endpoint = _get_endpoint()
    if not endpoint:
        return None

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": os.getenv("VLLM_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "user", "content": _build_classifier_prompt(query)}],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    try:
        r = requests.post(
            f"{endpoint}/chat/completions", headers=headers, json=payload, timeout=15,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        # Strip optional ```json fences
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        return json.loads(content)
    except Exception as e:
        print(f"[classifier] LLM classification failed: {e}")
        return None


def classify_query(query: str) -> ClassificationResult:
    """Classify a query into a query_type, with LLM-primary + heuristic fallback."""
    query = query.strip()
    if not query:
        return ClassificationResult("macro", [], "Empty query; defaulted to macro.", used_llm=False)

    if is_llm_available():
        parsed = _call_classifier_llm(query)
        if parsed and parsed.get("query_type") in VALID_QUERY_TYPES:
            return ClassificationResult(
                query_type=parsed["query_type"],
                tickers=[t.upper() for t in parsed.get("tickers", []) if isinstance(t, str)],
                reasoning=parsed.get("reasoning", "LLM classification"),
                used_llm=True,
            )
        # Fall through to heuristic on any failure (parse error, invalid type, LLM down)

    return _heuristic_classify(query)
