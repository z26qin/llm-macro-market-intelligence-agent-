"""Market data retrieval via yfinance."""

from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf

from utils.config import OIL_TICKERS, OIL_DISPLAY, NEOCLOUD_TICKERS, CRYPTO_TICKERS, AI_ROBOTICS_TICKERS, CREDIT_TICKERS


@dataclass
class PriceSnapshot:
    ticker: str
    name: str
    price: float | None
    change_1d_pct: float | None
    change_5d_pct: float | None
    error: str | None = None


@dataclass
class TechnicalSnapshot:
    ticker: str
    name: str
    price: float | None
    change_1d_pct: float | None
    change_5d_pct: float | None
    change_20d_pct: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    rvol: float | None
    error: str | None = None


@dataclass
class CreditSpread:
    name: str
    spread: float | None          # Current spread (price ratio or diff)
    spread_1d_change: float | None  # 1-day change in spread
    spread_5d_change: float | None  # 5-day change in spread
    interpretation: str           # "widening" / "tightening" / "stable"


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


def _bollinger_bands(series, window: int = 20, num_std: float = 2.0) -> tuple[float | None, float | None, float | None]:
    """Compute Bollinger Bands upper/middle/lower from close price series."""
    if series is None or len(series) < window:
        return None, None, None
    try:
        recent = series.iloc[-window:]
        mean = float(recent.mean())
        std = float(recent.std())
        return (
            round(mean + num_std * std, 2),
            round(mean, 2),
            round(mean - num_std * std, 2),
        )
    except Exception:
        return None, None, None


def _rsi(series, period: int = 14) -> float | None:
    """Compute RSI (Relative Strength Index) using Wilder's smoothing."""
    if series is None or len(series) < period + 1:
        return None
    try:
        delta = series.diff().dropna()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.iloc[:period].mean()
        avg_loss = losses.iloc[:period].mean()
        # Wilder smoothing for the remainder
        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    except Exception:
        return None


def _rvol(volume_series, window: int = 5) -> float | None:
    """Relative Volume: current volume / average volume over prior `window` trading days.

    The current day is excluded from the average — RVOL compares *today* against
    the recent baseline.
    """
    if volume_series is None or len(volume_series) < window + 1:
        return None
    try:
        current = float(volume_series.iloc[-1])
        prior_avg = float(volume_series.iloc[-(window + 1):-1].mean())
        if prior_avg <= 0:
            return None
        return round(current / prior_avg, 2)
    except Exception:
        return None


def _macd(series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    """Compute MACD, signal line, and histogram using exponential moving averages.

    Returns: (macd_line, signal_line, histogram) — latest values.
    """
    if series is None or len(series) < slow + signal:
        return None, None, None
    try:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return (
            round(float(macd_line.iloc[-1]), 3),
            round(float(signal_line.iloc[-1]), 3),
            round(float(hist.iloc[-1]), 3),
        )
    except Exception:
        return None, None, None


def get_technical_snapshot(ticker: str) -> TechnicalSnapshot:
    """Fetch price + technical indicators (1d/5d/20d returns, BB, RSI) for one ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")  # enough for 20d BB + RSI
        if hist.empty:
            return TechnicalSnapshot(
                ticker=ticker, name=ticker, price=None,
                change_1d_pct=None, change_5d_pct=None, change_20d_pct=None,
                bb_upper=None, bb_middle=None, bb_lower=None, rsi=None,
                macd=None, macd_signal=None, macd_hist=None,
                rvol=None,
                error="No price data available",
            )
        close = hist["Close"]
        price = round(float(close.iloc[-1]), 2)
        info = tk.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        bb_up, bb_mid, bb_lo = _bollinger_bands(close, window=20)
        macd_line, macd_sig, macd_h = _macd(close)
        rvol_val = _rvol(hist["Volume"], window=5) if "Volume" in hist.columns else None
        return TechnicalSnapshot(
            ticker=ticker,
            name=name,
            price=price,
            change_1d_pct=_pct_change(close, 1),
            change_5d_pct=_pct_change(close, 5),
            change_20d_pct=_pct_change(close, 20),
            bb_upper=bb_up,
            bb_middle=bb_mid,
            bb_lower=bb_lo,
            rsi=_rsi(close, period=14),
            macd=macd_line,
            macd_signal=macd_sig,
            macd_hist=macd_h,
            rvol=rvol_val,
        )
    except Exception as e:
        return TechnicalSnapshot(
            ticker=ticker, name=ticker, price=None,
            change_1d_pct=None, change_5d_pct=None, change_20d_pct=None,
            bb_upper=None, bb_middle=None, bb_lower=None, rsi=None,
            macd=None, macd_signal=None, macd_hist=None,
            rvol=None,
            error=str(e),
        )


def get_all_technical_snapshots() -> list[TechnicalSnapshot]:
    """Return technical snapshots for every tracked ticker across all categories."""
    tickers: list[str] = []
    seen: set[str] = set()
    for group in (OIL_TICKERS, NEOCLOUD_TICKERS, CRYPTO_TICKERS,
                  AI_ROBOTICS_TICKERS, CREDIT_TICKERS):
        for t in group:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return [get_technical_snapshot(t) for t in tickers]


def get_snapshots_for_query(query: str, query_type: str) -> list[PriceSnapshot]:
    """Return price snapshots relevant to the query type."""
    if query_type == "oil":
        return [get_price_snapshot(t) for t in OIL_TICKERS]
    elif query_type == "neocloud":
        return [get_price_snapshot(t) for t in NEOCLOUD_TICKERS]
    elif query_type == "crypto":
        return [get_price_snapshot(t) for t in CRYPTO_TICKERS]
    elif query_type == "ai_robotics":
        return [get_price_snapshot(t) for t in AI_ROBOTICS_TICKERS]
    elif query_type == "credit":
        return [get_price_snapshot(t) for t in CREDIT_TICKERS]
    elif query_type == "ticker":
        symbol = query.strip().upper()
        return [get_price_snapshot(symbol)]
    else:
        # Macro topic — show broad market proxies
        return [get_price_snapshot(t) for t in ["SPY", "QQQ", "TLT", "DX-Y.NYB"]]


def _calc_spread_ratio(ticker1: str, ticker2: str) -> tuple[float | None, float | None, float | None]:
    """Calculate spread ratio between two tickers and its changes."""
    try:
        t1 = yf.Ticker(ticker1)
        t2 = yf.Ticker(ticker2)
        h1 = t1.history(period="1mo")["Close"]
        h2 = t2.history(period="1mo")["Close"]

        if h1.empty or h2.empty or len(h1) < 6 or len(h2) < 6:
            return None, None, None

        # Current ratio
        current = float(h1.iloc[-1]) / float(h2.iloc[-1])

        # 1-day ago ratio
        ratio_1d_ago = float(h1.iloc[-2]) / float(h2.iloc[-2])
        change_1d = round((current - ratio_1d_ago) / ratio_1d_ago * 100, 3)

        # 5-day ago ratio
        ratio_5d_ago = float(h1.iloc[-6]) / float(h2.iloc[-6])
        change_5d = round((current - ratio_5d_ago) / ratio_5d_ago * 100, 3)

        return round(current, 4), change_1d, change_5d
    except Exception:
        return None, None, None


def get_credit_spreads() -> list[CreditSpread]:
    """Calculate key credit spread metrics."""
    spreads = []

    # HYG/TLT - High Yield vs Treasuries (risk appetite)
    hyg_tlt, hyg_tlt_1d, hyg_tlt_5d = _calc_spread_ratio("HYG", "TLT")
    if hyg_tlt is not None:
        interp = "tightening (risk-on)" if (hyg_tlt_1d or 0) > 0 else "widening (risk-off)" if (hyg_tlt_1d or 0) < 0 else "stable"
        spreads.append(CreditSpread(
            name="HYG/TLT (HY vs Treasury)",
            spread=hyg_tlt,
            spread_1d_change=hyg_tlt_1d,
            spread_5d_change=hyg_tlt_5d,
            interpretation=interp,
        ))

    # HYG/LQD - High Yield vs Investment Grade
    hyg_lqd, hyg_lqd_1d, hyg_lqd_5d = _calc_spread_ratio("HYG", "LQD")
    if hyg_lqd is not None:
        interp = "HY outperforming (risk-on)" if (hyg_lqd_1d or 0) > 0 else "HY underperforming (risk-off)" if (hyg_lqd_1d or 0) < 0 else "stable"
        spreads.append(CreditSpread(
            name="HYG/LQD (HY vs IG)",
            spread=hyg_lqd,
            spread_1d_change=hyg_lqd_1d,
            spread_5d_change=hyg_lqd_5d,
            interpretation=interp,
        ))

    # EMB/TLT - EM Credit vs Treasuries
    emb_tlt, emb_tlt_1d, emb_tlt_5d = _calc_spread_ratio("EMB", "TLT")
    if emb_tlt is not None:
        interp = "EM tightening" if (emb_tlt_1d or 0) > 0 else "EM widening" if (emb_tlt_1d or 0) < 0 else "stable"
        spreads.append(CreditSpread(
            name="EMB/TLT (EM vs Treasury)",
            spread=emb_tlt,
            spread_1d_change=emb_tlt_1d,
            spread_5d_change=emb_tlt_5d,
            interpretation=interp,
        ))

    # BKLN/LQD - Loans vs IG (liquidity stress)
    bkln_lqd, bkln_lqd_1d, bkln_lqd_5d = _calc_spread_ratio("BKLN", "LQD")
    if bkln_lqd is not None:
        interp = "loans outperforming" if (bkln_lqd_1d or 0) > 0 else "liquidity stress" if (bkln_lqd_1d or 0) < 0 else "stable"
        spreads.append(CreditSpread(
            name="BKLN/LQD (Loans vs IG)",
            spread=bkln_lqd,
            spread_1d_change=bkln_lqd_1d,
            spread_5d_change=bkln_lqd_5d,
            interpretation=interp,
        ))

    return spreads
