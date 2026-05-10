"""FRED (St. Louis Fed) macro data client with on-disk cache."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from utils.config import FRED_API_KEY, FRED_CACHE_DIR, FRED_CACHE_TTL_SECONDS


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


@dataclass
class FredSeries:
    series_id: str
    name: str
    units: str
    observations: list[tuple[str, float]] = field(default_factory=list)  # (date, value)
    error: str | None = None

    @property
    def latest(self) -> tuple[str, float] | None:
        return self.observations[-1] if self.observations else None

    @property
    def latest_value(self) -> float | None:
        last = self.latest
        return last[1] if last else None

    def change(self, periods: int = 1) -> float | None:
        """Absolute change vs N observations ago."""
        if len(self.observations) <= periods:
            return None
        return round(self.observations[-1][1] - self.observations[-1 - periods][1], 4)


# Macro series we care about by default. id → (display name, units)
DEFAULT_SERIES: dict[str, tuple[str, str]] = {
    "DGS10": ("US 10Y Treasury Yield", "%"),
    "DGS2": ("US 2Y Treasury Yield", "%"),
    "T10Y2Y": ("10Y-2Y Spread", "%"),
    "DFF": ("Fed Funds Rate (effective)", "%"),
    "DTWEXBGS": ("US Dollar Index (broad)", "index"),
    "VIXCLS": ("VIX", "index"),
    "DCOILWTICO": ("WTI Crude (FRED)", "$/bbl"),
    "BAMLH0A0HYM2": ("HY OAS (ICE BofA)", "%"),
    "T10YIE": ("10Y Breakeven Inflation", "%"),
    "UNRATE": ("Unemployment Rate", "%"),
    "CPIAUCSL": ("CPI (All Urban)", "index"),
}


def _cache_path(series_id: str) -> Path:
    p = Path(FRED_CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{series_id}.json"


def _load_cache(series_id: str) -> dict | None:
    path = _cache_path(series_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - payload.get("fetched_at", 0) > FRED_CACHE_TTL_SECONDS:
        return None
    return payload


def _save_cache(series_id: str, observations: list[tuple[str, float]]) -> None:
    path = _cache_path(series_id)
    try:
        path.write_text(json.dumps({
            "fetched_at": time.time(),
            "observations": observations,
        }))
    except OSError:
        pass


def fetch_series(series_id: str, *, limit: int = 60) -> FredSeries:
    """Fetch a FRED series, using cache if fresh.

    `limit` controls how many recent observations to return (sorted ascending by date).
    """
    name, units = DEFAULT_SERIES.get(series_id, (series_id, ""))

    cached = _load_cache(series_id)
    if cached is not None:
        obs = [tuple(o) for o in cached["observations"]][-limit:]
        return FredSeries(series_id=series_id, name=name, units=units, observations=obs)

    if not FRED_API_KEY:
        return FredSeries(
            series_id=series_id, name=name, units=units,
            error="FRED_API_KEY not set",
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return FredSeries(series_id=series_id, name=name, units=units, error=str(e))

    raw = data.get("observations", [])
    parsed: list[tuple[str, float]] = []
    for row in raw:
        val = row.get("value")
        if val in (None, ".", ""):
            continue
        try:
            parsed.append((row["date"], float(val)))
        except (ValueError, KeyError):
            continue
    parsed.sort(key=lambda t: t[0])  # ascending
    _save_cache(series_id, parsed)
    return FredSeries(series_id=series_id, name=name, units=units, observations=parsed)


def fetch_default_panel() -> list[FredSeries]:
    """Fetch the standard macro dashboard set."""
    return [fetch_series(sid) for sid in DEFAULT_SERIES]


def macro_summary_lines() -> list[str]:
    """Compact one-line-per-series summary for prompt injection."""
    lines: list[str] = []
    for s in fetch_default_panel():
        if s.error or not s.latest:
            lines.append(f"- {s.name} ({s.series_id}): unavailable")
            continue
        date, val = s.latest
        chg5 = s.change(5)
        chg20 = s.change(20)
        parts = [f"{val:.2f}{s.units}"]
        if chg5 is not None:
            parts.append(f"5d Δ {chg5:+.2f}")
        if chg20 is not None:
            parts.append(f"20d Δ {chg20:+.2f}")
        lines.append(f"- {s.name} ({s.series_id}, {date}): {', '.join(parts)}")
    return lines
