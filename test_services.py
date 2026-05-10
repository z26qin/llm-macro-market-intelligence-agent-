"""Offline tests for the data-service layer.

Mocks all network and yfinance access so tests are deterministic and run in
under a second. Style matches test_agent_loop.py (no pytest dependency).

    python3 test_services.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Use throwaway cache dirs so tests never collide with real .cache contents
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="services_test_"))
os.environ["FRED_CACHE_DIR"] = str(_TMP_ROOT / "fred")
os.environ["COT_CACHE_DIR"] = str(_TMP_ROOT / "cot")
os.environ["FRED_API_KEY"] = "test_key"


# ─────────────────────────────────────────────────────────────────────────────
# FRED
# ─────────────────────────────────────────────────────────────────────────────

def _fake_fred_response(observations: list[tuple[str, str]]) -> MagicMock:
    """Build a fake requests.Response for the FRED observations endpoint."""
    resp = MagicMock()
    resp.json.return_value = {
        "observations": [{"date": d, "value": v} for d, v in observations],
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_fred_basic_fetch_and_change():
    import services.fred as fred
    # Reload config-derived path
    fred.FRED_CACHE_DIR = os.environ["FRED_CACHE_DIR"]

    obs = [("2026-04-01", "4.30"), ("2026-04-15", "4.40"), ("2026-05-01", "4.45")]
    with patch.object(fred.requests, "get",
                       return_value=_fake_fred_response(obs)) as mock_get:
        s = fred.fetch_series("DGS10")

    assert mock_get.call_count == 1
    assert s.error is None
    assert len(s.observations) == 3
    assert s.latest == ("2026-05-01", 4.45)
    assert s.change(1) == round(4.45 - 4.40, 4)
    assert s.change(2) == round(4.45 - 4.30, 4)
    assert s.change(99) is None  # not enough history


def test_fred_uses_cache_on_second_call():
    import services.fred as fred
    obs = [("2026-05-01", "4.45")]
    fake = _fake_fred_response(obs)
    with patch.object(fred.requests, "get", return_value=fake) as mock_get:
        fred.fetch_series("DGS2")     # network call
        fred.fetch_series("DGS2")     # should hit cache
    assert mock_get.call_count == 1, "second call should not have hit network"


def test_fred_no_key_returns_error():
    import services.fred as fred
    saved = fred.FRED_API_KEY
    try:
        fred.FRED_API_KEY = ""
        # Use a unique series id so cache from earlier tests doesn't shadow
        s = fred.fetch_series("CPIAUCSL")
        assert s.error == "FRED_API_KEY not set"
        assert s.latest is None
    finally:
        fred.FRED_API_KEY = saved


def test_fred_skips_missing_observations():
    import services.fred as fred
    obs_with_holes = [("2026-04-01", "."), ("2026-04-15", "1.0"),
                       ("2026-04-22", ""), ("2026-05-01", "2.0")]
    with patch.object(fred.requests, "get",
                       return_value=_fake_fred_response(obs_with_holes)):
        s = fred.fetch_series("UNRATE")
    assert len(s.observations) == 2
    assert s.observations[0] == ("2026-04-15", 1.0)
    assert s.observations[1] == ("2026-05-01", 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# COT
# ─────────────────────────────────────────────────────────────────────────────

def _fake_cot_rows(market_name: str, weeks: int, dataset: str = "legacy") -> MagicMock:
    """Newest-first rows simulating what Socrata returns."""
    rows = []
    for i in range(weeks):
        # net = long - short. Make it grow week over week so z is positive.
        long_v = 100_000 + i * 1_000
        short_v = 50_000
        date = f"2026-{(5 - (i // 4)):02d}-{(28 - (i % 4) * 7):02d}"
        if dataset == "legacy":
            row = {
                "market_and_exchange_names": market_name,
                "report_date_as_yyyy_mm_dd": f"{date}T00:00:00.000",
                "noncomm_positions_long_all": str(long_v),
                "noncomm_positions_short_all": str(short_v),
            }
        else:  # tff
            row = {
                "market_and_exchange_names": market_name,
                "report_date_as_yyyy_mm_dd": f"{date}T00:00:00.000",
                "lev_money_positions_long": str(long_v),
                "lev_money_positions_short": str(short_v),
            }
        rows.append(row)
    resp = MagicMock()
    resp.json.return_value = rows
    resp.raise_for_status = MagicMock()
    return resp


def test_cot_legacy_market_returns_snapshot_with_zscore():
    import services.cot as cot
    cot.COT_CACHE_DIR = os.environ["COT_CACHE_DIR"]

    fake = _fake_cot_rows("WTI-PHYSICAL - NEW YORK MERCANTILE", weeks=52, dataset="legacy")
    with patch.object(cot.requests, "get", return_value=fake):
        s = cot.fetch_cot("CL")

    assert s.error is None
    assert s.code == "CL"
    assert s.name == "WTI Crude"
    # In the fixture, i=0 is the newest date (long=100k, short=50k → net=50k).
    # Older rows i=1..51 have larger long values, so the trailing mean > latest
    # → z_score should be NEGATIVE (latest is below the historical mean).
    assert s.net_noncomm == 50_000
    assert s.week_change is not None
    assert s.z_score is not None and s.z_score < 0
    assert len(s.history_net) == 52


def test_cot_tff_uses_leveraged_money_fields():
    import services.cot as cot
    fake = _fake_cot_rows("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                            weeks=8, dataset="tff")
    with patch.object(cot.requests, "get", return_value=fake):
        s = cot.fetch_cot("ES")
    assert s.error is None
    # Should have parsed leveraged-fund fields, not noncomm (which are absent)
    assert s.net_noncomm is not None and s.net_noncomm > 0


def test_cot_unknown_market_errors():
    import services.cot as cot
    s = cot.fetch_cot("XX")
    assert "unknown market code" in (s.error or "")


def test_cot_uses_cache_on_second_call():
    import services.cot as cot
    fake = _fake_cot_rows("GOLD - COMMODITY EXCHANGE INC", weeks=10, dataset="legacy")
    with patch.object(cot.requests, "get", return_value=fake) as mock_get:
        cot.fetch_cot("GC")
        cot.fetch_cot("GC")
    assert mock_get.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

def _fake_yf_ticker(spot: float, expiries: tuple, atm_iv_call: float = 0.18,
                     atm_iv_put: float = 0.20):
    """Build a fake yfinance.Ticker covering everything our service touches."""
    import pandas as pd
    fake = MagicMock()
    fake.fast_info = {"lastPrice": spot}
    fake.options = expiries
    chain = MagicMock()
    chain.calls = pd.DataFrame([
        {"strike": spot - 5, "impliedVolatility": atm_iv_call - 0.03},
        {"strike": spot,     "impliedVolatility": atm_iv_call},
        {"strike": spot + 5, "impliedVolatility": atm_iv_call + 0.03},
    ])
    chain.puts = pd.DataFrame([
        {"strike": spot - 5, "impliedVolatility": atm_iv_put + 0.03},
        {"strike": spot,     "impliedVolatility": atm_iv_put},
        {"strike": spot + 5, "impliedVolatility": atm_iv_put - 0.03},
    ])
    fake.option_chain = MagicMock(return_value=chain)
    return fake


def test_options_iv_basic():
    import services.options as options
    from datetime import date, timedelta
    # Pick an expiry well past the 21d threshold
    far = (date.today() + timedelta(days=30)).isoformat()
    near = (date.today() + timedelta(days=3)).isoformat()  # too close, must skip
    fake = _fake_yf_ticker(spot=100.0, expiries=(near, far),
                             atm_iv_call=0.22, atm_iv_put=0.24)
    with patch.object(options.yf, "Ticker", return_value=fake):
        snap = options.get_iv_snapshot("FAKE")
    assert snap.error is None
    assert snap.spot == 100.0
    assert snap.expiry == far
    assert snap.days_to_expiry == 30
    assert snap.call_iv == 0.22
    assert snap.put_iv == 0.24
    assert snap.avg_iv == round((0.22 + 0.24) / 2, 4)


def test_options_iv_no_listed_options():
    import services.options as options
    fake = MagicMock()
    fake.fast_info = {"lastPrice": 100.0}
    fake.options = ()  # nothing listed
    with patch.object(options.yf, "Ticker", return_value=fake):
        snap = options.get_iv_snapshot("BTC-USD")
    assert snap.error == "no listed options"
    assert snap.spot == 100.0
    assert snap.avg_iv is None


def test_options_iv_zero_iv_filtered():
    """yfinance returns 0 IV for stale strikes — service should not let those leak."""
    import services.options as options
    from datetime import date, timedelta
    far = (date.today() + timedelta(days=30)).isoformat()
    fake = _fake_yf_ticker(spot=100.0, expiries=(far,),
                             atm_iv_call=0.0, atm_iv_put=0.20)
    with patch.object(options.yf, "Ticker", return_value=fake):
        snap = options.get_iv_snapshot("FAKE")
    assert snap.call_iv is None
    assert snap.put_iv == 0.20
    assert snap.avg_iv == 0.20  # average of just the put


# ─────────────────────────────────────────────────────────────────────────────
# Tools registry
# ─────────────────────────────────────────────────────────────────────────────

def test_tools_registry_has_expected_tools():
    from services import tools
    expected = {
        "search_news", "get_prices", "analyze_sentiment",
        "get_credit_spreads", "get_fear_greed",
        "get_macro_series", "get_macro_panel",
        "get_positioning", "get_positioning_panel",
        "get_options_iv", "finalize",
    }
    assert set(tools.names()) >= expected


def test_tools_schemas_are_well_formed():
    from services import tools
    for s in tools.schemas():
        assert s["type"] == "function"
        f = s["function"]
        assert isinstance(f["name"], str) and f["name"]
        assert isinstance(f["description"], str) and f["description"]
        assert isinstance(f["parameters"], dict)
        assert f["parameters"].get("type") == "object"


def test_tools_finalize_writes_rationale_to_collected():
    from services import tools
    collected = {}
    out = tools.get("finalize").impl(collected, rationale="enough evidence")
    assert collected["_finalize_rationale"] == "enough evidence"
    assert out["acknowledged"] is True


def test_tools_get_macro_series_dispatches_to_fred(monkeypatch=None):
    from services import tools
    import services.fred as fred

    class Fake:
        series_id = "DGS10"
        name = "US 10Y Treasury Yield"
        units = "%"
        latest = ("2026-05-01", 4.45)
        observations = [("2026-04-01", 4.30), ("2026-05-01", 4.45)]
        error = None

        def change(self, n):
            return 0.15 if n == 1 else None

    with patch.object(fred, "fetch_series", return_value=Fake()):
        # the tool imports fred_fetch_series inside services/tools.py;
        # patch THAT alias, not services.fred.fetch_series.
        with patch.object(tools, "fred_fetch_series", return_value=Fake()):
            collected = {}
            out = tools.get("get_macro_series").impl(collected, series_id="DGS10")

    assert out["series_id"] == "DGS10"
    assert out["latest_value"] == 4.45
    assert "macro" in collected
    assert "DGS10" in collected["macro"]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_fred_basic_fetch_and_change,
    test_fred_uses_cache_on_second_call,
    test_fred_no_key_returns_error,
    test_fred_skips_missing_observations,
    test_cot_legacy_market_returns_snapshot_with_zscore,
    test_cot_tff_uses_leveraged_money_fields,
    test_cot_unknown_market_errors,
    test_cot_uses_cache_on_second_call,
    test_options_iv_basic,
    test_options_iv_no_listed_options,
    test_options_iv_zero_iv_filtered,
    test_tools_registry_has_expected_tools,
    test_tools_schemas_are_well_formed,
    test_tools_finalize_writes_rationale_to_collected,
    test_tools_get_macro_series_dispatches_to_fred,
]


def main() -> int:
    failures = []
    try:
        for fn in TESTS:
            try:
                print(f"→ {fn.__name__}")
                fn()
                print(f"  ✓ pass")
            except AssertionError as e:
                print(f"  ✗ FAIL: {e}")
                failures.append((fn.__name__, str(e)))
            except Exception as e:
                print(f"  ✗ ERROR: {type(e).__name__}: {e}")
                failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)

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
