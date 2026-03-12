"""Sentiment analysis using FinBERT or a lightweight mock fallback."""

from __future__ import annotations

from dataclasses import dataclass
from utils.config import SENTIMENT_MODE


@dataclass
class SentimentResult:
    label: str          # positive / negative / neutral
    score: float        # confidence 0-1
    text: str           # input text


@dataclass
class SentimentSummary:
    avg_score: float
    positive: int
    negative: int
    neutral: int
    details: list[SentimentResult]
    mode: str           # finbert or mock


# ── FinBERT loader (lazy singleton) ──────────────────────────────────────────

_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
    return _finbert_pipeline


def _run_finbert(texts: list[str]) -> list[SentimentResult]:
    pipe = _load_finbert()
    results: list[SentimentResult] = []
    for text in texts:
        truncated = text[:512]
        preds = pipe(truncated)[0]  # list of {label, score}
        best = max(preds, key=lambda p: p["score"])
        results.append(SentimentResult(
            label=best["label"],
            score=round(best["score"], 3),
            text=truncated[:120],
        ))
    return results


# ── Mock fallback ────────────────────────────────────────────────────────────

def _run_mock(texts: list[str]) -> list[SentimentResult]:
    """Keyword-based mock sentiment for quick prototyping."""
    pos_words = {"surge", "rally", "beat", "upgrade", "record", "gain", "bull", "growth", "strong"}
    neg_words = {"drop", "crash", "miss", "downgrade", "fall", "bear", "weak", "loss", "cut", "risk"}

    results: list[SentimentResult] = []
    for text in texts:
        lower = text.lower()
        pos = sum(1 for w in pos_words if w in lower)
        neg = sum(1 for w in neg_words if w in lower)
        if pos > neg:
            label, score = "positive", 0.7
        elif neg > pos:
            label, score = "negative", 0.7
        else:
            label, score = "neutral", 0.5
        results.append(SentimentResult(label=label, score=score, text=text[:120]))
    return results


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_sentiment(texts: list[str]) -> SentimentSummary:
    """Score a list of texts and return an aggregated summary."""
    if not texts:
        return SentimentSummary(
            avg_score=0, positive=0, negative=0, neutral=0, details=[], mode="none",
        )

    mode = SENTIMENT_MODE
    if mode == "finbert":
        try:
            details = _run_finbert(texts)
        except Exception as e:
            print(f"[sentiment] FinBERT failed, falling back to mock: {e}")
            details = _run_mock(texts)
            mode = "mock"
    else:
        details = _run_mock(texts)
        mode = "mock"

    pos = sum(1 for d in details if d.label == "positive")
    neg = sum(1 for d in details if d.label == "negative")
    neu = sum(1 for d in details if d.label == "neutral")

    # Signed average: positive -> +score, negative -> -score
    signed = []
    for d in details:
        if d.label == "positive":
            signed.append(d.score)
        elif d.label == "negative":
            signed.append(-d.score)
        else:
            signed.append(0.0)
    avg = round(sum(signed) / len(signed), 3) if signed else 0.0

    return SentimentSummary(
        avg_score=avg, positive=pos, negative=neg, neutral=neu,
        details=details, mode=mode,
    )
