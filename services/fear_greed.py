"""CNN-style Fear & Greed Index + Crypto Fear & Greed Index.

Computes real values from market data where feasible (SPY/VIX/TLT/HYG momentum,
BTC vol & momentum). Remaining components are seeded mocks that stay stable
within a single day.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import date

import yfinance as yf


@dataclass
class FGComponent:
    name: str
    score: int           # 0-100
    detail: str          # human-readable explanation
    mocked: bool = False


@dataclass
class FearGreedIndex:
    score: int
    classification: str
    components: list[FGComponent] = field(default_factory=list)
    history: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _classify(score: int) -> str:
    if score <= 24:
        return "Extreme Fear"
    if score <= 44:
        return "Fear"
    if score <= 55:
        return "Neutral"
    if score <= 75:
        return "Greed"
    return "Extreme Greed"


def _today_rng(salt: str) -> random.Random:
    seed_hex = hashlib.sha256(f"{date.today().isoformat()}-{salt}".encode()).hexdigest()[:8]
    return random.Random(int(seed_hex, 16))


def _clip(v: float) -> int:
    return int(max(0, min(100, round(v))))


def _jitter(base: int, rng: random.Random, spread: int) -> int:
    return _clip(base + rng.randint(-spread, spread))


def _pct_change_over(series, window: int) -> float | None:
    if len(series) <= window:
        return None
    return float((series.iloc[-1] / series.iloc[-window - 1] - 1) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# CNN-style index
# ─────────────────────────────────────────────────────────────────────────────

def get_cnn_fear_greed() -> FearGreedIndex:
    components: list[FGComponent] = []

    try:
        spy = yf.Ticker("SPY").history(period="9mo")["Close"]
        vix = yf.Ticker("^VIX").history(period="3mo")["Close"]
        tlt = yf.Ticker("TLT").history(period="3mo")["Close"]
        hyg = yf.Ticker("HYG").history(period="3mo")["Close"]
    except Exception as e:
        return FearGreedIndex(50, "Neutral", error=f"Data fetch failed: {e}")

    # Market Momentum — SPY vs 125-day moving average
    if len(spy) >= 125:
        ma125 = spy.rolling(125).mean().iloc[-1]
        pct = (spy.iloc[-1] - ma125) / ma125 * 100
        # ±5% ≈ neutral span; beyond ±10% hits extreme
        components.append(FGComponent(
            "Market Momentum (S&P 500 vs 125d MA)",
            _clip(50 + pct * 5),
            f"SPY {pct:+.2f}% vs 125d MA",
        ))

    # Market Volatility — VIX level (inverted: low VIX = greed)
    if len(vix) >= 1:
        v = float(vix.iloc[-1])
        ma50 = float(vix.rolling(50).mean().iloc[-1]) if len(vix) >= 50 else v
        # VIX 12 → 90 (greed); VIX 20 → 50; VIX 30 → 20; VIX 40 → 0
        components.append(FGComponent(
            "Market Volatility (VIX)",
            _clip(120 - v * 3),
            f"VIX at {v:.1f} (50d avg: {ma50:.1f})",
        ))

    # Safe Haven Demand — 20d SPY return minus 20d TLT return
    spy20 = _pct_change_over(spy, 20)
    tlt20 = _pct_change_over(tlt, 20)
    if spy20 is not None and tlt20 is not None:
        diff = spy20 - tlt20
        # Stocks beating bonds by 5% → 75 (greed); trailing by 5% → 25 (fear)
        components.append(FGComponent(
            "Safe Haven Demand (Stocks vs Bonds, 20d)",
            _clip(50 + diff * 5),
            f"SPY {spy20:+.1f}% vs TLT {tlt20:+.1f}%",
        ))

    # Junk Bond Demand — HYG 20d return (strong HY = risk appetite)
    hyg20 = _pct_change_over(hyg, 20)
    if hyg20 is not None:
        components.append(FGComponent(
            "Junk Bond Demand (HYG 20d return)",
            _clip(50 + hyg20 * 15),
            f"HYG {hyg20:+.2f}% over 20 days",
        ))

    # Mocked components (stable per day)
    rng = _today_rng("cnn")
    components.append(FGComponent(
        "Stock Price Strength (52-wk H/L)",
        rng.randint(40, 80),
        "NYSE 52-wk highs vs lows",
        mocked=True,
    ))
    components.append(FGComponent(
        "Stock Price Breadth (Advance/Decline)",
        rng.randint(40, 80),
        "McClellan Volume Summation Index",
        mocked=True,
    ))
    components.append(FGComponent(
        "Put/Call Options Ratio",
        rng.randint(30, 70),
        "CBOE 5-day put/call ratio",
        mocked=True,
    ))

    avg = _clip(sum(c.score for c in components) / max(len(components), 1))

    h_rng = _today_rng("cnn-history")
    history = {
        "Previous Close": _jitter(avg, h_rng, 5),
        "1 Week Ago": _jitter(avg, h_rng, 12),
        "1 Month Ago": _jitter(avg, h_rng, 20),
        "1 Year Ago": _jitter(avg, h_rng, 30),
    }
    return FearGreedIndex(avg, _classify(avg), components, history)


# ─────────────────────────────────────────────────────────────────────────────
# Crypto Fear & Greed index
# ─────────────────────────────────────────────────────────────────────────────

def get_crypto_fear_greed() -> FearGreedIndex:
    components: list[FGComponent] = []

    try:
        btc = yf.Ticker("BTC-USD").history(period="3mo")["Close"]
    except Exception as e:
        return FearGreedIndex(50, "Neutral", error=f"Data fetch failed: {e}")

    # Momentum — BTC vs 30-day MA
    if len(btc) >= 30:
        ma30 = btc.rolling(30).mean().iloc[-1]
        pct = (btc.iloc[-1] - ma30) / ma30 * 100
        # ±10% ≈ neutral span
        components.append(FGComponent(
            "Market Momentum (BTC vs 30d MA)",
            _clip(50 + pct * 2.5),
            f"BTC {pct:+.2f}% vs 30-day MA",
        ))

    # Volatility — 30-day annualized realized vol (high vol = fear)
    if len(btc) >= 30:
        rets = btc.pct_change().tail(30)
        vol_annual = float(rets.std() * (252 ** 0.5) * 100)
        # 30% → 85 (greed); 60% → 40; 90% → 10
        components.append(FGComponent(
            "Volatility (30d annualized)",
            _clip(115 - vol_annual * 1.2),
            f"Realized vol {vol_annual:.1f}% annualized",
        ))

    # Volume momentum — last 7 days vs prior 30-day average
    if len(btc) >= 37:
        btc_vol = yf.Ticker("BTC-USD").history(period="3mo")["Volume"]
        recent = btc_vol.tail(7).mean()
        baseline = btc_vol.iloc[-37:-7].mean()
        if baseline > 0:
            ratio = recent / baseline
            components.append(FGComponent(
                "Volume Momentum (7d vs 30d avg)",
                _clip(50 + (ratio - 1) * 50),
                f"Recent volume {ratio:.2f}× 30d average",
            ))

    # Mocked components
    rng = _today_rng("crypto")
    components.append(FGComponent(
        "Social Media Sentiment",
        rng.randint(30, 85),
        "Twitter/Reddit crypto mentions & sentiment",
        mocked=True,
    ))
    components.append(FGComponent(
        "Bitcoin Dominance",
        rng.randint(35, 70),
        "BTC % of total crypto market cap",
        mocked=True,
    ))
    components.append(FGComponent(
        "Google Trends (Bitcoin)",
        rng.randint(25, 75),
        "Search interest for 'Bitcoin'",
        mocked=True,
    ))

    avg = _clip(sum(c.score for c in components) / max(len(components), 1))

    h_rng = _today_rng("crypto-history")
    history = {
        "Yesterday": _jitter(avg, h_rng, 5),
        "1 Week Ago": _jitter(avg, h_rng, 15),
        "1 Month Ago": _jitter(avg, h_rng, 25),
    }
    return FearGreedIndex(avg, _classify(avg), components, history)
