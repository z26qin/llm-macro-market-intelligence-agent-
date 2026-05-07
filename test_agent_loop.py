"""Offline tests for the ReAct agent loop.

Mocks vLLM (so no GPU/endpoint needed) and stubs out tool implementations,
verifying that the loop logic, duplicate-call guard, stop reasons, and trace
persistence all work end-to-end. Run as a script:

    python3 test_agent_loop.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

# Ensure is_llm_available() returns True before importing the agent
os.environ.setdefault("VLLM_ENDPOINT", "http://mock:8000")

import services.agent as agent_mod
from services.agent import run_agent
from services import tools as tool_registry


# ─────────────────────────────────────────────────────────────────────────────
# vLLM response helpers — shape what the agent sees as if from chat/completions
# ─────────────────────────────────────────────────────────────────────────────

def _msg(content=None, tool_calls=None):
    out = {"role": "assistant", "content": content or ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _tc(name: str, args: dict, id_: str = "c") -> dict:
    return {
        "id": id_, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _resp(content=None, tool_calls=None) -> dict:
    return {"choices": [{"message": _msg(content, tool_calls)}]}


@dataclass
class FakeValidation:
    passed: bool = True
    confidence_score: float = 80.0
    errors: list = None
    warnings: list = None
    attempts: int = 1


def _fake_narrative(*_args, **_kwargs):
    return ("MOCKED NARRATIVE.", FakeValidation(errors=[], warnings=[]))


# Stub each registered tool so no network calls happen during tests
def _stub_tools():
    """Replace every tool's impl with a no-op that records and returns a stub."""
    originals = {}
    for name, tool in list(tool_registry._REGISTRY.items()):
        originals[name] = tool.impl

        def make_stub(tool_name):
            def stub(collected, **kwargs):
                # Mark in collected so consumers know stubs ran
                collected.setdefault("_stub_calls", []).append((tool_name, kwargs))
                return {"stub": tool_name, "args": kwargs}
            return stub

        tool.impl = make_stub(name)
    return originals


def _restore_tools(originals):
    for name, impl in originals.items():
        tool_registry._REGISTRY[name].impl = impl


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_script(script: list[dict], query: str = "NVDA",
                      query_type_hint: str = "ticker", trace_dir: str | None = None):
    """Run the agent with a scripted vLLM response sequence."""
    seq = list(script)

    def fake_vllm(_messages):
        if not seq:
            raise AssertionError("agent kept iterating past the end of the script")
        return seq.pop(0)

    originals = _stub_tools()
    try:
        with patch.object(agent_mod, "_call_vllm_with_tools", fake_vllm), \
             patch.object(agent_mod, "generate_and_validate_narrative", _fake_narrative):
            if trace_dir:
                old = agent_mod.TRACE_DIR
                agent_mod.TRACE_DIR = trace_dir
                try:
                    return run_agent(query, query_type_hint=query_type_hint)
                finally:
                    agent_mod.TRACE_DIR = old
            return run_agent(query, query_type_hint=query_type_hint)
    finally:
        _restore_tools(originals)


def test_happy_path_finalizes(tmp_dir: Path):
    script = [
        _resp(tool_calls=[_tc("get_prices", {"query": "NVDA", "query_type": "ticker"}, "c1")]),
        _resp(tool_calls=[_tc("search_news", {"query": "NVDA earnings", "query_type": "ticker"}, "c2")]),
        _resp(tool_calls=[_tc("get_macro_panel", {}, "c3")]),
        _resp(tool_calls=[_tc("get_options_iv", {"ticker": "NVDA"}, "c4")]),
        _resp(tool_calls=[_tc("finalize", {"rationale": "Sufficient evidence."}, "c5")]),
    ]
    result = _run_with_script(script, trace_dir=str(tmp_dir))

    assert result.stop_reason == "finalized", f"got {result.stop_reason}"
    assert result.iterations == 5
    assert [t.tool for t in result.trace] == [
        "get_prices", "search_news", "get_macro_panel", "get_options_iv", "finalize",
    ]
    assert result.rationale == "Sufficient evidence."
    assert result.narrative == "MOCKED NARRATIVE."

    # Trace persisted
    assert result.trace_path and Path(result.trace_path).exists()
    saved = json.loads(Path(result.trace_path).read_text())
    assert saved["query"] == "NVDA"
    assert saved["stop_reason"] == "finalized"
    assert len(saved["trace"]) == 5
    assert saved["narrative"] == "MOCKED NARRATIVE."
    print(f"  trace at {result.trace_path}")


def test_no_more_tools_stops(tmp_dir: Path):
    """LLM returns content without a tool call — loop should stop cleanly."""
    script = [
        _resp(tool_calls=[_tc("get_prices", {"query": "X", "query_type": "ticker"}, "c1")]),
        _resp(content="I have enough."),  # no tool calls
    ]
    result = _run_with_script(script, trace_dir=str(tmp_dir))
    assert result.stop_reason == "no_more_tools"
    assert result.iterations == 2
    assert [t.tool for t in result.trace] == ["get_prices"]


def test_duplicate_call_refused(tmp_dir: Path):
    """Identical tool args twice → second call must return the refusal stub."""
    args = {"query": "oil", "query_type": "oil"}
    script = [
        _resp(tool_calls=[_tc("get_prices", args, "c1")]),
        _resp(tool_calls=[_tc("get_prices", args, "c2")]),  # duplicate
        _resp(tool_calls=[_tc("finalize", {"rationale": "ok"}, "c3")]),
    ]
    result = _run_with_script(script, query="oil", query_type_hint="oil",
                                trace_dir=str(tmp_dir))
    assert result.stop_reason == "finalized"
    assert len(result.trace) == 3
    # Second get_prices must show the refusal
    second = result.trace[1]
    assert second.tool == "get_prices"
    assert "duplicate call refused" in second.summary, second.summary


def test_max_iters_capped(tmp_dir: Path):
    """If LLM keeps calling tools forever, agent stops at MAX_AGENT_ITERATIONS."""
    # Generate distinct calls so duplicate guard doesn't fire
    script = [
        _resp(tool_calls=[_tc("get_macro_series", {"series_id": f"S{i}"}, f"c{i}")])
        for i in range(20)  # more than the cap
    ]
    result = _run_with_script(script, trace_dir=str(tmp_dir))
    assert result.stop_reason == "max_iters"
    assert result.iterations == agent_mod.MAX_AGENT_ITERATIONS  # 10
    assert len(result.trace) == agent_mod.MAX_AGENT_ITERATIONS


def test_unknown_tool_returns_error(tmp_dir: Path):
    script = [
        _resp(tool_calls=[_tc("nonexistent_tool", {}, "c1")]),
        _resp(tool_calls=[_tc("finalize", {"rationale": "stop"}, "c2")]),
    ]
    result = _run_with_script(script, trace_dir=str(tmp_dir))
    assert result.stop_reason == "finalized"
    assert "unknown tool" in result.trace[0].summary


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_happy_path_finalizes,
    test_no_more_tools_stops,
    test_duplicate_call_refused,
    test_max_iters_capped,
    test_unknown_tool_returns_error,
]


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="agent_test_"))
    failures = []
    try:
        for fn in TESTS:
            sub = tmp_root / fn.__name__
            sub.mkdir()
            try:
                print(f"→ {fn.__name__}")
                fn(sub)
                print(f"  ✓ pass")
            except AssertionError as e:
                print(f"  ✗ FAIL: {e}")
                failures.append((fn.__name__, str(e)))
            except Exception as e:
                print(f"  ✗ ERROR: {type(e).__name__}: {e}")
                failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} tests failed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    print(f"all {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
