"""CFTC Commitments of Traders (COT) positioning data with on-disk cache.

Pulls the legacy futures-only report from publicreporting.cftc.gov (Socrata, no
API key required). Net non-commercial position = long − short for the
speculative crowd. Z-score is computed vs trailing 52-week mean/std so the
prompt can flag crowded longs (>+2σ) or shorts (<−2σ).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from utils.config import COT_CACHE_DIR, COT_CACHE_TTL_SECONDS


LEGACY_FUTURES_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
TFF_FUTURES_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
HISTORY_WEEKS = 52


# code → (CFTC market_and_exchange_names substring, display name, dataset)
# - "legacy" report: speculator = non-commercial. Used for physical commodities + VIX.
# - "tff" report: speculator = leveraged funds. Used for index futures, rates, FX.
MARKETS: dict[str, tuple[str, str, str]] = {
    "CL":  ("WTI-PHYSICAL",                  "WTI Crude",         "legacy"),
    "GC":  ("GOLD - COMMODITY EXCHANGE",     "Gold",              "legacy"),
    "HG":  ("COPPER- #1",                    "Copper",            "legacy"),
    "ES":  ("E-MINI S&P 500",                "E-mini S&P 500",    "tff"),
    "NQ":  ("NASDAQ-100 Consolidated",       "Nasdaq 100",        "tff"),
    "6E":  ("EURO FX",                       "Euro FX",           "tff"),
    "VX":  ("VIX FUTURES",                   "VIX",               "legacy"),
    # Note: U.S. Treasury futures positioning is not available via CFTC's public
    # Socrata feed past Feb 2022. Use FRED yields (DGS10/DGS2) for rates context.
}


# Fields differ between legacy and TFF datasets. Speculator = noncomm in legacy
# (broad speculator bucket), and leveraged funds in TFF (hedge funds).
SPEC_FIELDS = {
    "legacy": ("noncomm_positions_long_all",   "noncomm_positions_short_all"),
    "tff":    ("lev_money_positions_long",     "lev_money_positions_short"),
}

DATASET_URLS = {
    "legacy": LEGACY_FUTURES_URL,
    "tff":    TFF_FUTURES_URL,
}


@dataclass
class CotSnapshot:
    code: str
    name: str
    report_date: str | None = None
    long_noncomm: int | None = None
    short_noncomm: int | None = None
    net_noncomm: int | None = None
    week_change: int | None = None     # net change vs prior week
    z_score: float | None = None       # net vs trailing 52w
    history_net: list[tuple[str, int]] = field(default_factory=list)  # ascending
    error: str | None = None


def _cache_path(code: str) -> Path:
    p = Path(COT_CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{code}.json"


def _load_cache(code: str) -> list[dict] | None:
    path = _cache_path(code)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - payload.get("fetched_at", 0) > COT_CACHE_TTL_SECONDS:
        return None
    return payload.get("rows", [])


def _save_cache(code: str, rows: list[dict]) -> None:
    try:
        _cache_path(code).write_text(json.dumps({
            "fetched_at": time.time(),
            "rows": rows,
        }))
    except OSError:
        pass


def _query_dataset(dataset: str, market_substring: str) -> list[dict]:
    """Fetch up to HISTORY_WEEKS rows for a market substring, newest first."""
    where = f"market_and_exchange_names like '%{market_substring}%'"
    params = {
        "$where": where,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": HISTORY_WEEKS,
    }
    resp = requests.get(DATASET_URLS[dataset], params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _to_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def fetch_cot(code: str) -> CotSnapshot:
    """Fetch latest COT snapshot for a tracked market."""
    if code not in MARKETS:
        return CotSnapshot(code=code, name=code, error=f"unknown market code {code}")
    substring, name, dataset = MARKETS[code]
    long_field, short_field = SPEC_FIELDS[dataset]

    rows = _load_cache(code)
    if rows is None:
        try:
            rows = _query_dataset(dataset, substring)
        except requests.RequestException as e:
            return CotSnapshot(code=code, name=name, error=str(e))
        _save_cache(code, rows)

    if not rows:
        return CotSnapshot(code=code, name=name, error="no rows returned")

    # rows are newest-first from the API
    parsed_history: list[tuple[str, int]] = []
    for r in rows:
        date = r.get("report_date_as_yyyy_mm_dd", "")[:10]
        long_v = _to_int(r.get(long_field))
        short_v = _to_int(r.get(short_field))
        if long_v is None or short_v is None or not date:
            continue
        parsed_history.append((date, long_v - short_v))

    parsed_history.sort(key=lambda t: t[0])  # ascending

    if not parsed_history:
        return CotSnapshot(code=code, name=name, error="failed to parse rows")

    latest_row = rows[0]
    latest_long = _to_int(latest_row.get(long_field))
    latest_short = _to_int(latest_row.get(short_field))
    latest_date = latest_row.get("report_date_as_yyyy_mm_dd", "")[:10]
    latest_net = parsed_history[-1][1]

    week_change = None
    if len(parsed_history) >= 2:
        week_change = latest_net - parsed_history[-2][1]

    z = None
    if len(parsed_history) >= 8:
        nets = [n for _, n in parsed_history]
        mean = statistics.mean(nets)
        # population stdev guards against tiny n; sample is fine here too
        try:
            stdev = statistics.stdev(nets)
        except statistics.StatisticsError:
            stdev = 0.0
        if stdev > 0:
            z = round((latest_net - mean) / stdev, 2)

    return CotSnapshot(
        code=code, name=name,
        report_date=latest_date,
        long_noncomm=latest_long, short_noncomm=latest_short,
        net_noncomm=latest_net, week_change=week_change,
        z_score=z, history_net=parsed_history,
    )


def fetch_default_panel() -> list[CotSnapshot]:
    return [fetch_cot(c) for c in MARKETS]


def positioning_summary_lines() -> list[str]:
    """Compact one-line-per-market summary for prompt injection."""
    lines: list[str] = []
    for s in fetch_default_panel():
        if s.error or s.net_noncomm is None:
            lines.append(f"- {s.name} ({s.code}): unavailable")
            continue
        wc = f", Δw {s.week_change:+,}" if s.week_change is not None else ""
        z = f", z={s.z_score:+.2f}" if s.z_score is not None else ""
        flag = ""
        if s.z_score is not None:
            if s.z_score >= 2.0:
                flag = " ⚠ crowded long"
            elif s.z_score <= -2.0:
                flag = " ⚠ crowded short"
        lines.append(
            f"- {s.name} ({s.code}, {s.report_date}): "
            f"net non-comm {s.net_noncomm:+,}{wc}{z}{flag}"
        )
    return lines
