"""Market data retrieval via yfinance."""

from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf

from utils.config import OIL_TICKERS, OIL_DISPLAY, NEOCLOUD_TICKERS


@dataclass
class PriceSnapshot:
    ticker: str
    name: str
    price: float | None
    change_1d_pct: float | None
    change_5d_pct: float | None
    error: str | None = None


def _pct_change(series, periods: int) -> float | None:
    """Compute percentage change over the last N periods in a price series."""
    if series is None or len(series) < periods + 1:
        return None
    try:
        old = float(series.iloc[-(periods + 1)])
        new = float(series.iloc[-1])
        if old == 0:
            return None
        return round((new - old) / old * 100, 2)
    except Exception:
        return None


def get_price_snapshot(ticker: str) -> PriceSnapshot:
    """Fetch latest price and short-term moves for a single ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1mo")
        if hist.empty:
            return PriceSnapshot(
                ticker=ticker, name=ticker, price=None,
                change_1d_pct=None, change_5d_pct=None,
                error="No price data available",
            )
        close = hist["Close"]
        price = round(float(close.iloc[-1]), 2)
        info = tk.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        return PriceSnapshot(
            ticker=ticker,
            name=name,
            price=price,
            change_1d_pct=_pct_change(close, 1),
            change_5d_pct=_pct_change(close, 5),
        )
    except Exception as e:
        return PriceSnapshot(
            ticker=ticker, name=ticker, price=None,
            change_1d_pct=None, change_5d_pct=None,
            error=str(e),
        )


def get_snapshots_for_query(query: str, query_type: str) -> list[PriceSnapshot]:
    """Return price snapshots relevant to the query type."""
    if query_type == "oil":
        return [get_price_snapshot(t) for t in OIL_TICKERS]
    elif query_type == "neocloud":
        return [get_price_snapshot(t) for t in NEOCLOUD_TICKERS]
    elif query_type == "ticker":
        symbol = query.strip().upper()
        return [get_price_snapshot(symbol)]
    else:
        # Macro topic — show broad market proxies
        return [get_price_snapshot(t) for t in ["SPY", "QQQ", "TLT", "DX-Y.NYB"]]
