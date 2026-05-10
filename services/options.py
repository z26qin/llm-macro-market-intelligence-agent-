"""Options implied volatility snapshot via yfinance.

Picks the nearest expiry ≥ MIN_DAYS_TO_EXPIRY (default 21) as the "30d" proxy
and averages ATM call + ATM put IVs. Skips contracts with stale or zero quotes
to avoid noise from illiquid strikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date

import yfinance as yf


MIN_DAYS_TO_EXPIRY = 21


@dataclass
class OptionsIvSnapshot:
    ticker: str
    spot: float | None = None
    expiry: str | None = None         # YYYY-MM-DD
    days_to_expiry: int | None = None
    atm_strike: float | None = None
    call_iv: float | None = None      # decimal, e.g. 0.18 = 18%
    put_iv: float | None = None
    avg_iv: float | None = None       # mean of call + put IV
    error: str | None = None


def _spot_from_ticker(t: yf.Ticker) -> float | None:
    try:
        v = t.fast_info["lastPrice"]
        return float(v) if v else None
    except Exception:
        pass
    try:
        hist = t.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _pick_expiry(expiries: tuple[str, ...]) -> str | None:
    today = _date.today()
    best: tuple[str, int] | None = None
    for e in expiries:
        try:
            dt = _date.fromisoformat(e)
        except ValueError:
            continue
        days = (dt - today).days
        if days < MIN_DAYS_TO_EXPIRY:
            continue
        if best is None or days < best[1]:
            best = (e, days)
    return best[0] if best else None


def get_iv_snapshot(ticker: str) -> OptionsIvSnapshot:
    snap = OptionsIvSnapshot(ticker=ticker.upper())
    try:
        t = yf.Ticker(ticker)
        spot = _spot_from_ticker(t)
        if spot is None:
            snap.error = "no spot price"
            return snap
        snap.spot = round(spot, 4)

        expiries = t.options
        if not expiries:
            snap.error = "no listed options"
            return snap

        expiry = _pick_expiry(expiries)
        if expiry is None:
            # fall back to the furthest available
            expiry = expiries[-1]
        snap.expiry = expiry
        snap.days_to_expiry = (_date.fromisoformat(expiry) - _date.today()).days

        chain = t.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            snap.error = "empty option chain"
            return snap

        ci = (calls["strike"] - spot).abs().idxmin()
        pi = (puts["strike"] - spot).abs().idxmin()
        call_strike = float(calls.loc[ci, "strike"])
        put_strike = float(puts.loc[pi, "strike"])
        snap.atm_strike = call_strike  # both should match for liquid names

        call_iv = float(calls.loc[ci, "impliedVolatility"]) if "impliedVolatility" in calls.columns else None
        put_iv = float(puts.loc[pi, "impliedVolatility"]) if "impliedVolatility" in puts.columns else None
        # filter zeros / NaNs that yfinance returns for stale quotes
        snap.call_iv = call_iv if call_iv and call_iv > 0 else None
        snap.put_iv = put_iv if put_iv and put_iv > 0 else None

        ivs = [iv for iv in (snap.call_iv, snap.put_iv) if iv is not None]
        if ivs:
            snap.avg_iv = round(sum(ivs) / len(ivs), 4)
        else:
            snap.error = "no valid IV in chain"
    except Exception as e:
        snap.error = f"{type(e).__name__}: {e}"
    return snap
