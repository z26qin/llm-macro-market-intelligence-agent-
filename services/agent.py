"""Tool-calling ReAct agent.

The LLM dynamically picks which tools to invoke for a given query instead of
running the fixed pipeline. Uses vLLM's OpenAI-compatible tool-calling API.

Tools live in `services.tools` (single registry, schema + impl per tool).

Requires:
- VLLM_ENDPOINT set
- The served model to support tool/function calling (e.g. Llama-3.1-Instruct
  started with `--enable-auto-tool-choice --tool-call-parser llama3_json`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

from services.llm import (
    DEFAULT_MODEL, _get_endpoint, is_llm_available,
    generate_and_validate_narrative,
)
from services.narrative import generate_narrative
from services.sentiment import analyze_sentiment
from services import tools as tool_registry


MAX_AGENT_ITERATIONS = 10  # was 6 — more headroom for multi-tool exploration
TRACE_DIR = ".cache/agent_traces"


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentTrace:
    tool: str
    args: dict[str, Any]
    summary: str       # short human-readable preview of the result
    duration_ms: int = 0


@dataclass
class AgentResult:
    narrative: str
    validation: Any
    trace: list[AgentTrace]
    collected: dict[str, Any]
    query_type: str
    iterations: int
    stop_reason: str   # "finalized" | "max_iters" | "no_more_tools" | "llm_error: ..."
    rationale: str = ""
    trace_path: str | None = None  # where the trace JSON was persisted


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — encourages a real plan→act→observe→reflect cycle
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a macro market intelligence agent operating a tool-calling ReAct loop.

For each query, run a deliberate plan → act → observe → reflect cycle:
1. PLAN: pick the next single tool that closes the largest gap in your evidence.
2. ACT: emit ONE tool call.
3. OBSERVE: read the tool result.
4. REFLECT: decide whether the evidence is sufficient or another tool is needed.
5. Repeat until evidence is sufficient, then call `finalize`.

Tool selection guidance:
- `get_prices` and `search_news` are usually the first two calls.
- `analyze_sentiment` runs on the headline TITLES returned by `search_news`.
- `get_macro_panel` for any macro-flavored question (rates, dollar, vol, credit).
  Prefer the panel; fall back to `get_macro_series` if you need exactly one series.
- `get_positioning_panel` to scan for crowded extremes (|z|≥2). Or `get_positioning`
  for a single market. Especially useful for commodities (CL, GC, HG) and equity
  indices (ES, NQ).
- `get_options_iv` to gauge expected vol on a single ticker. Compare to VIX or
  realized vol when relevant.
- `get_credit_spreads` only for credit/risk-appetite queries.
- `get_fear_greed` only for broad regime/mood queries.

Hard rules:
- Do NOT repeat a tool call with identical arguments.
- One tool call per turn; do not batch.
- Cap: aim for 3-6 tool calls before `finalize`. Hard ceiling is 10 iterations.
- Call `finalize` with a one-sentence rationale once evidence is sufficient.
"""


# ─────────────────────────────────────────────────────────────────────────────
# vLLM tool-calling
# ─────────────────────────────────────────────────────────────────────────────

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
        "tools": tool_registry.schemas(),
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


# ─────────────────────────────────────────────────────────────────────────────
# Trace persistence
# ─────────────────────────────────────────────────────────────────────────────

def _persist_trace(query: str, query_type: str, messages: list[dict],
                    trace: list[AgentTrace], result: AgentResult) -> str:
    Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
    qhash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = Path(TRACE_DIR) / f"{ts}_{query_type}_{qhash}.json"
    payload = {
        "query": query,
        "query_type": query_type,
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "rationale": result.rationale,
        "trace": [asdict(t) for t in trace],
        "messages": _redact_messages(messages),
        "validation": {
            "passed": getattr(result.validation, "passed", None),
            "confidence_score": getattr(result.validation, "confidence_score", None),
            "errors": getattr(result.validation, "errors", []),
            "warnings": getattr(result.validation, "warnings", []),
            "attempts": getattr(result.validation, "attempts", None),
        },
        "narrative": result.narrative,
    }
    try:
        path.write_text(json.dumps(payload, default=str, indent=2))
        return str(path)
    except OSError as e:
        print(f"[agent] failed to persist trace: {e}")
        return ""


def _redact_messages(messages: list[dict]) -> list[dict]:
    """Drop oversized tool payloads from persisted messages — keep first 600 chars."""
    out = []
    for m in messages:
        copy = dict(m)
        content = copy.get("content")
        if isinstance(content, str) and len(content) > 600:
            copy["content"] = content[:600] + f"…(+{len(content)-600} chars)"
        out.append(copy)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loop
# ─────────────────────────────────────────────────────────────────────────────

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
    iteration = 0

    seen_calls: set[tuple[str, str]] = set()

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

            # Guard: refuse to repeat the exact same call
            sig = (name, json.dumps(args, sort_keys=True, default=str))
            if sig in seen_calls and name != "finalize":
                result = {"error": "duplicate call refused; pick a different tool or finalize"}
                duration_ms = 0
            else:
                seen_calls.add(sig)
                tool = tool_registry.get(name)
                start = time.time()
                if tool is None:
                    result = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        result = tool.impl(collected, **args)
                    except TypeError as e:
                        result = {"error": f"bad arguments: {e}"}
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                duration_ms = int((time.time() - start) * 1000)

            trace.append(AgentTrace(tool=name, args=args,
                                      summary=_truncate(result), duration_ms=duration_ms))
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

    narrative, validation = generate_and_validate_narrative(
        topic=query,
        query_type=collected.get("query_type", query_type_hint),
        results=collected.get("results", []),
        snapshots=collected.get("snapshots", []),
        sentiment=collected.get("sentiment") or analyze_sentiment([]),
        template_fallback_fn=generate_narrative,
    )

    result = AgentResult(
        narrative=narrative, validation=validation, trace=trace,
        collected=collected,
        query_type=collected.get("query_type", query_type_hint),
        iterations=iteration, stop_reason=stop_reason, rationale=rationale,
    )
    result.trace_path = _persist_trace(query, result.query_type, messages, trace, result)
    return result
