"""Portfolio construction: value time-series given positions entered on 2026-01-02."""

from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf
import pandas as pd


ENTRY_DATE = "2026-01-02"  # First trading day of 2026 (Jan 1 is a holiday)


@dataclass
class PositionResult:
    ticker: str
    shares: float
    entry_price: float
    current_price: float | None
    invested: float
    current_value: float | None
    pnl: float | None
    error: str | None = None


def compute_portfolio(positions: list[dict], total_capital: float) -> dict:
    """Compute portfolio value time-series and per-position summary.

    Args:
        positions: list of {"ticker": str, "entry_price": float, "shares": float}
        total_capital: Total capital (USD). Cash = total_capital − sum(shares × entry_price).

    Returns:
        {
            "dates": [ISO date strings],
            "total_values": [float] aligned to dates,
            "positions": list[PositionResult],
            "cash": float,
            "invested": float,
            "error": str | None,
        }
    """
    valid = [p for p in positions if p.get("ticker") and p.get("shares") and p.get("entry_price")]
    if not valid:
        return {"error": "No valid positions entered.", "dates": [], "total_values": [],
                "positions": [], "cash": total_capital, "invested": 0.0}

    invested = sum(float(p["shares"]) * float(p["entry_price"]) for p in valid)
    cash = total_capital - invested

    # Fetch price history for all tickers since entry date
    tickers = [p["ticker"].strip().upper() for p in valid]
    price_frames: dict[str, pd.Series] = {}
    position_results: list[PositionResult] = []

    for pos in valid:
        tk = pos["ticker"].strip().upper()
        shares = float(pos["shares"])
        entry_px = float(pos["entry_price"])
        try:
            hist = yf.Ticker(tk).history(start=ENTRY_DATE)
            if hist.empty:
                position_results.append(PositionResult(
                    ticker=tk, shares=shares, entry_price=entry_px,
                    current_price=None, invested=shares * entry_px,
                    current_value=None, pnl=None,
                    error="No price data",
                ))
                continue
            close = hist["Close"]
            # Normalize index to date (strip tz) for joining across tickers
            close.index = close.index.tz_localize(None).normalize() if close.index.tz is not None else close.index.normalize()
            price_frames[tk] = close
            current_px = round(float(close.iloc[-1]), 2)
            position_results.append(PositionResult(
                ticker=tk, shares=shares, entry_price=entry_px,
                current_price=current_px,
                invested=shares * entry_px,
                current_value=shares * current_px,
                pnl=shares * (current_px - entry_px),
            ))
        except Exception as e:
            position_results.append(PositionResult(
                ticker=tk, shares=shares, entry_price=entry_px,
                current_price=None, invested=shares * entry_px,
                current_value=None, pnl=None,
                error=str(e),
            ))

    # Build unified time series: outer-join all price series, forward-fill gaps
    if not price_frames:
        return {"error": "No price data for any position.", "dates": [], "total_values": [],
                "positions": position_results, "cash": cash, "invested": invested}

    df = pd.concat(price_frames, axis=1).ffill()

    # Total position value at each date = Σ(shares_i × price_i)
    shares_map = {p["ticker"].strip().upper(): float(p["shares"]) for p in valid}
    position_values = sum(df[tk] * shares_map[tk] for tk in df.columns if tk in shares_map)
    total_values = position_values + cash

    dates = [d.strftime("%Y-%m-%d") for d in total_values.index]
    values = [round(float(v), 2) for v in total_values.values]

    return {
        "error": None,
        "dates": dates,
        "total_values": values,
        "positions": position_results,
        "cash": round(cash, 2),
        "invested": round(invested, 2),
    }
