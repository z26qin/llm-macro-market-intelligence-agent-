"""All `_render_*` functions extracted from app.py. Pure presentation."""

from __future__ import annotations

from datetime import datetime

from dash import dcc, html
import plotly.graph_objects as go

from services.search import SearchResult
from services.market_data import (
    PriceSnapshot, CreditSpread, TechnicalSnapshot,
)
from services.sentiment import SentimentSummary
from services.validation import ValidationResult
from services.classifier import ClassificationResult
from services.agent import AgentResult
from services.fear_greed import FearGreedIndex
from services.fred import FredSeries
from services.cot import CotSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Common utilities
# ─────────────────────────────────────────────────────────────────────────────

def _format_date(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return date_str[:10] if date_str else ""


def _truncate_arg(v, limit: int = 30) -> str:
    s = str(v)
    return s if len(s) <= limit else s[:limit - 3] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_headlines(results: list[SearchResult]) -> html.Div:
    items = []
    for r in results:
        link = html.A(r.title, href=r.url, target="_blank",
                       style={"color": "#1a0dab"}) if r.url else html.Span(r.title)
        date_display = _format_date(r.published_date)
        date_span = html.Span(f" [{date_display}]", style={"fontSize": "11px", "color": "#888"}) if date_display else None
        items.append(html.Li([
            link,
            date_span,
            html.Br(),
            html.Span(r.snippet[:200], style={"fontSize": "12px", "color": "#555"})
        ]))
    return html.Div([
        html.H4("Latest Headlines"),
        html.Ul(items, style={"paddingLeft": "20px", "lineHeight": "1.7"}),
    ])


def _render_prices(snapshots: list[PriceSnapshot]) -> html.Div:
    rows = []
    for s in snapshots:
        if s.error:
            rows.append(html.Tr([html.Td(s.ticker), html.Td(s.error, colSpan=3,
                        style={"color": "red"})]))
        else:
            d1 = f"{s.change_1d_pct:+.2f}%" if s.change_1d_pct is not None else "—"
            d5 = f"{s.change_5d_pct:+.2f}%" if s.change_5d_pct is not None else "—"
            color = "#080" if (s.change_1d_pct or 0) >= 0 else "#b00"
            rows.append(html.Tr([
                html.Td(f"{s.name} ({s.ticker})", style={"fontWeight": "bold"}),
                html.Td(f"${s.price}"),
                html.Td(d1, style={"color": color}),
                html.Td(d5),
            ]))
    header = html.Tr([html.Th("Asset"), html.Th("Price"), html.Th("1d"), html.Th("5d")])
    return html.Div([
        html.H4("Price Snapshot"),
        html.Table([header] + rows,
                   style={"borderCollapse": "collapse", "width": "100%",
                          "fontSize": "13px", "textAlign": "left"}),
    ], style={"overflowX": "auto"})


def _render_credit_spreads(spreads: list[CreditSpread]) -> html.Div:
    if not spreads:
        return html.Div()

    rows = []
    for sp in spreads:
        d1 = f"{sp.spread_1d_change:+.3f}%" if sp.spread_1d_change is not None else "—"
        d5 = f"{sp.spread_5d_change:+.3f}%" if sp.spread_5d_change is not None else "—"
        if "tightening" in sp.interpretation or "outperforming" in sp.interpretation or "risk-on" in sp.interpretation:
            color = "#080"
        elif "widening" in sp.interpretation or "underperforming" in sp.interpretation or "risk-off" in sp.interpretation or "stress" in sp.interpretation:
            color = "#b00"
        else:
            color = "#888"
        rows.append(html.Tr([
            html.Td(sp.name, style={"fontWeight": "bold"}),
            html.Td(f"{sp.spread:.4f}" if sp.spread else "—"),
            html.Td(d1, style={"color": color}),
            html.Td(d5),
            html.Td(sp.interpretation, style={"fontSize": "11px", "color": color}),
        ]))

    header = html.Tr([
        html.Th("Spread"), html.Th("Value"), html.Th("1d Δ"),
        html.Th("5d Δ"), html.Th("Interpretation"),
    ])
    return html.Div([
        html.H4("Credit Spreads & Risk Appetite"),
        html.Table([header] + rows,
                   style={"borderCollapse": "collapse", "width": "100%",
                          "fontSize": "13px", "textAlign": "left"}),
        html.P("↑ Ratio = credit outperforming (spreads tightening, risk-on)",
               style={"fontSize": "11px", "color": "#666", "marginTop": "8px"}),
        html.P("↓ Ratio = credit underperforming (spreads widening, risk-off)",
               style={"fontSize": "11px", "color": "#666"}),
    ], style={"overflowX": "auto", "marginTop": "16px"})


def _render_sentiment(sentiment: SentimentSummary) -> html.Div:
    bar_color = "#080" if sentiment.avg_score > 0 else "#b00" if sentiment.avg_score < 0 else "#888"
    return html.Div([
        html.H4("Sentiment Summary"),
        html.P([
            html.Span(f"Avg: {sentiment.avg_score:+.3f}  ",
                       style={"color": bar_color, "fontWeight": "bold"}),
            f"+{sentiment.positive} / -{sentiment.negative} / ~{sentiment.neutral}  ",
            html.Span(f"(mode: {sentiment.mode})", style={"fontSize": "12px", "color": "#777"}),
        ]),
    ])


def _render_narrative(narrative: str) -> html.Div:
    return html.Div([
        html.H4("Generated Narrative"),
        dcc.Markdown(narrative, style={"backgroundColor": "#f8f8f8", "padding": "16px",
                                        "borderRadius": "6px", "fontSize": "13px",
                                        "lineHeight": "1.6", "whiteSpace": "pre-wrap"}),
    ])


def _render_validation(validation: ValidationResult) -> html.Div:
    if validation.passed and validation.confidence_score >= 80:
        status_color = "#080"
        status_text = "✓ VERIFIED"
    elif validation.passed and validation.confidence_score >= 60:
        status_color = "#f90"
        status_text = "⚠ VERIFIED (Medium Confidence)"
    elif validation.passed:
        status_color = "#f90"
        status_text = "⚠ VERIFIED (Low Confidence)"
    else:
        status_color = "#b00"
        status_text = "✗ VALIDATION FAILED"

    details_items = []
    details_items.append(html.Div([
        html.Span("Overall Confidence: ", style={"fontWeight": "bold"}),
        html.Span(f"{validation.confidence_score:.0f}/100",
                 style={"color": status_color, "fontWeight": "bold", "fontSize": "16px"}),
    ], style={"marginBottom": "8px"}))

    attempts = getattr(validation, "attempts", 1)
    if attempts > 1:
        details_items.append(html.Div([
            html.Span("Self-correction: ", style={"fontWeight": "bold"}),
            html.Span(f"{attempts} attempts", style={"color": "#06c", "fontWeight": "bold"}),
            html.Span(" — agent re-prompted after validator feedback",
                     style={"fontSize": "11px", "color": "#666", "marginLeft": "6px"}),
        ], style={"marginBottom": "8px"}))

    num_ver = validation.numerical_verification
    details_items.append(html.Div([
        html.Span("Numerical Claims: ", style={"fontWeight": "bold"}),
        html.Span(f"{num_ver['verified_claims']}/{num_ver['total_claims']} verified"),
        html.Span(f" ({num_ver['verification_rate']*100:.0f}%)",
                 style={"fontSize": "11px", "color": "#666"}),
    ], style={"marginBottom": "4px"}))

    cite_ver = validation.citation_verification
    details_items.append(html.Div([
        html.Span("Citations: ", style={"fontWeight": "bold"}),
        html.Span(f"{cite_ver['unique_citations']} sources cited"),
        html.Span(f" ({cite_ver['citation_coverage']*100:.0f}% coverage)",
                 style={"fontSize": "11px", "color": "#666"}),
    ], style={"marginBottom": "4px"}))

    if validation.errors:
        error_list = html.Ul([html.Li(err, style={"color": "#b00"}) for err in validation.errors],
                            style={"marginLeft": "16px", "fontSize": "12px"})
        details_items.append(html.Div([
            html.Span("❌ Errors:", style={"fontWeight": "bold", "color": "#b00"}),
            error_list,
        ], style={"marginTop": "8px"}))

    if validation.warnings:
        warning_list = html.Ul([html.Li(warn, style={"color": "#f90"}) for warn in validation.warnings],
                              style={"marginLeft": "16px", "fontSize": "12px"})
        details_items.append(html.Div([
            html.Span("⚠️ Warnings:", style={"fontWeight": "bold", "color": "#f90"}),
            warning_list,
        ], style={"marginTop": "8px"}))

    return html.Div([
        html.H4("Anti-Hallucination Validation",
               style={"display": "flex", "alignItems": "center", "gap": "12px"}),
        html.Div([
            html.Span(status_text, style={"color": status_color, "fontWeight": "bold",
                                          "fontSize": "14px", "padding": "4px 12px",
                                          "backgroundColor": f"{status_color}22",
                                          "borderRadius": "4px", "border": f"1px solid {status_color}"}),
        ], style={"marginBottom": "12px"}),
        html.Div(details_items,
                style={"fontSize": "13px", "backgroundColor": "#f8f8f8",
                       "padding": "12px", "borderRadius": "6px"}),
    ])


def _render_classification(c: ClassificationResult) -> html.Div:
    badge_color = "#06c" if c.used_llm else "#888"
    badge_text = "LLM-classified" if c.used_llm else "heuristic fallback"
    tickers = ", ".join(c.tickers) if c.tickers else "—"
    return html.Div([
        html.Div([
            html.Span("🤖 Query Classifier: ", style={"fontWeight": "bold"}),
            html.Span(f"{c.query_type}", style={
                "color": "#fff", "backgroundColor": badge_color,
                "padding": "2px 8px", "borderRadius": "3px",
                "fontWeight": "bold", "fontSize": "12px",
            }),
            html.Span(f"  ({badge_text})", style={"fontSize": "11px", "color": "#888"}),
        ]),
        html.Div([
            html.Span("Tickers extracted: ", style={"fontWeight": "bold", "fontSize": "12px"}),
            html.Span(tickers, style={"fontSize": "12px", "fontFamily": "Menlo, monospace"}),
        ], style={"marginTop": "4px"}),
        html.Div([
            html.Span("Reasoning: ", style={"fontWeight": "bold", "fontSize": "12px"}),
            html.Span(c.reasoning, style={"fontSize": "12px", "color": "#555"}),
        ], style={"marginTop": "4px"}),
    ], style={"padding": "10px 14px", "backgroundColor": "#f0f6ff",
              "borderLeft": "3px solid #06c", "borderRadius": "4px"})


def _render_debug(results: list[SearchResult], sentiment: SentimentSummary):
    return html.Details([
        html.Summary("Debug / Retrieved Evidence",
                     style={"cursor": "pointer", "fontSize": "13px",
                            "fontWeight": "bold", "marginTop": "12px"}),
        html.Pre(
            "\n".join(
                [f"[{d.label} {d.score:.2f}] {d.text}" for d in sentiment.details]
                + ["", "--- Raw search titles ---"]
                + [r.title for r in results]
            ),
            style={"fontSize": "11px", "backgroundColor": "#f0f0f0", "padding": "12px",
                   "maxHeight": "300px", "overflowY": "auto"},
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Agent panels
# ─────────────────────────────────────────────────────────────────────────────

def _render_agent_macro(collected: dict) -> html.Div:
    panel = collected.get("macro_panel") or list((collected.get("macro") or {}).values())
    rows = []
    for s in panel:
        if getattr(s, "error", None) or not getattr(s, "latest", None):
            rows.append(html.Tr([
                html.Td(f"{s.name} ({s.series_id})", style={"fontWeight": "bold"}),
                html.Td(s.error or "no data", colSpan=3,
                         style={"color": "#999", "fontStyle": "italic"}),
            ]))
            continue
        date, val = s.latest
        chg5 = s.change(5)
        chg5_str = f"{chg5:+.2f}" if chg5 is not None else "—"
        c5 = "#080" if (chg5 or 0) >= 0 else "#b00"
        rows.append(html.Tr([
            html.Td(f"{s.name} ({s.series_id})", style={"fontWeight": "bold"}),
            html.Td(f"{val:.2f} {s.units}".strip()),
            html.Td(date, style={"fontSize": "11px", "color": "#666"}),
            html.Td(chg5_str, style={"color": c5}),
        ]))
    header = html.Tr([html.Th("Series"), html.Th("Latest"), html.Th("As of"), html.Th("5d Δ")])
    return html.Div([
        html.H4("Macro Backdrop (FRED)"),
        html.Table([header] + rows,
                    style={"borderCollapse": "collapse", "width": "100%",
                           "fontSize": "13px", "textAlign": "left"}),
    ], style={"overflowX": "auto"})


def _render_agent_positioning(collected: dict) -> html.Div:
    panel = collected.get("positioning_panel") or list((collected.get("positioning") or {}).values())
    rows = []
    for s in panel:
        if getattr(s, "error", None) or s.net_noncomm is None:
            rows.append(html.Tr([
                html.Td(f"{s.name} ({s.code})", style={"fontWeight": "bold"}),
                html.Td(s.error or "no data", colSpan=3,
                         style={"color": "#999", "fontStyle": "italic"}),
            ]))
            continue
        z_str = f"{s.z_score:+.2f}" if s.z_score is not None else "—"
        flag = ""
        z_color = "#666"
        if s.z_score is not None and s.z_score >= 2.0:
            z_color, flag = "#b00", " crowded long"
        elif s.z_score is not None and s.z_score <= -2.0:
            z_color, flag = "#b00", " crowded short"
        rows.append(html.Tr([
            html.Td(f"{s.name} ({s.code})", style={"fontWeight": "bold"}),
            html.Td(s.report_date, style={"fontSize": "11px", "color": "#666"}),
            html.Td(f"{s.net_noncomm:+,}"),
            html.Td([z_str, html.Span(flag, style={"fontSize": "11px"})],
                     style={"color": z_color, "fontWeight": "bold" if flag else "normal"}),
        ]))
    header = html.Tr([html.Th("Market"), html.Th("As of"),
                       html.Th("Net spec"), html.Th("z (52w)")])
    return html.Div([
        html.H4("Speculator Positioning (CFTC COT)"),
        html.Table([header] + rows,
                    style={"borderCollapse": "collapse", "width": "100%",
                           "fontSize": "13px", "textAlign": "left"}),
    ], style={"overflowX": "auto"})


def _render_agent_options_iv(iv_map: dict) -> html.Div:
    rows = []
    for ticker, s in iv_map.items():
        if getattr(s, "error", None):
            rows.append(html.Tr([
                html.Td(ticker, style={"fontWeight": "bold"}),
                html.Td(s.error, colSpan=4,
                         style={"color": "#999", "fontStyle": "italic"}),
            ]))
            continue
        avg = f"{s.avg_iv*100:.1f}%" if s.avg_iv is not None else "—"
        rows.append(html.Tr([
            html.Td(ticker, style={"fontWeight": "bold"}),
            html.Td(f"${s.spot:.2f}" if s.spot else "—"),
            html.Td(s.expiry or "—"),
            html.Td(f"{s.days_to_expiry}d" if s.days_to_expiry is not None else "—"),
            html.Td(avg),
        ]))
    header = html.Tr([html.Th("Ticker"), html.Th("Spot"), html.Th("Expiry"),
                       html.Th("DTE"), html.Th("ATM IV (avg)")])
    return html.Div([
        html.H4("Options Implied Volatility"),
        html.Table([header] + rows,
                    style={"borderCollapse": "collapse", "width": "100%",
                           "fontSize": "13px", "textAlign": "left"}),
    ], style={"overflowX": "auto"})


def _render_live_trace(trace_so_far: list, stage: str) -> html.Div:
    items = []
    for i, step in enumerate(trace_so_far, 1):
        items.append(html.Div([
            html.Span(f"{i}. ", style={"color": "#888"}),
            html.Span(step.tool, style={"fontWeight": "bold", "color": "#06c",
                                          "fontFamily": "Menlo, monospace"}),
            html.Span(f"  ({step.duration_ms} ms)",
                       style={"fontSize": "11px", "color": "#888",
                              "fontFamily": "Menlo, monospace"}),
            html.Div(f"→ {step.summary}",
                     style={"fontSize": "11px", "color": "#666", "marginLeft": "20px",
                            "fontFamily": "Menlo, monospace"}),
        ], style={"padding": "2px 0"}))

    if stage == "narrating":
        items.append(html.Div("✎ generating narrative…",
                                style={"padding": "4px 0", "color": "#06c",
                                       "fontStyle": "italic", "fontFamily": "Menlo, monospace"}))
    elif stage == "done":
        items.append(html.Div("✓ done", style={"padding": "4px 0", "color": "#080",
                                                  "fontFamily": "Menlo, monospace"}))
    else:
        items.append(html.Div("⏱ thinking…",
                                style={"padding": "4px 0", "color": "#888",
                                       "fontStyle": "italic", "fontFamily": "Menlo, monospace"}))

    return html.Div([
        html.Div("🤖 Agent activity", style={"fontWeight": "bold", "fontSize": "13px",
                                                "marginBottom": "6px"}),
        html.Div(items, style={"backgroundColor": "#fafafa", "padding": "10px 14px",
                                 "borderRadius": "4px", "border": "1px solid #eee"}),
    ], style={"marginBottom": "12px"})


def _render_agent_trace(agent_result: AgentResult) -> html.Div:
    reason_color = {
        "finalized": "#080",
        "no_more_tools": "#c80",
        "max_iters": "#b00",
    }.get(agent_result.stop_reason, "#b00")

    trace_items = []
    for i, step in enumerate(agent_result.trace, 1):
        trace_items.append(html.Div([
            html.Span(f"{i}. ", style={"color": "#888"}),
            html.Span(step.tool, style={"fontWeight": "bold", "color": "#06c",
                                         "fontFamily": "Menlo, monospace"}),
            html.Span(f"({', '.join(f'{k}={_truncate_arg(v)}' for k, v in step.args.items())})",
                     style={"fontSize": "11px", "color": "#555",
                            "fontFamily": "Menlo, monospace"}),
            html.Div(f"→ {step.summary}",
                    style={"fontSize": "11px", "color": "#666", "marginLeft": "20px",
                           "fontFamily": "Menlo, monospace"}),
        ], style={"padding": "4px 0", "borderBottom": "1px dashed #eee"}))

    header = html.Div([
        html.Span("🤖 Agent trace: ", style={"fontWeight": "bold"}),
        html.Span(f"{agent_result.iterations} iterations, "
                 f"{len(agent_result.trace)} tool calls",
                 style={"fontSize": "12px", "color": "#555"}),
        html.Span(f"  [stop: {agent_result.stop_reason}]",
                 style={"color": reason_color, "fontWeight": "bold", "fontSize": "11px"}),
    ])

    rationale = html.Div([
        html.Span("Agent rationale: ", style={"fontWeight": "bold", "fontSize": "12px"}),
        html.Span(agent_result.rationale or "—", style={"fontSize": "12px", "color": "#555"}),
    ], style={"marginTop": "4px"}) if agent_result.rationale else None

    return html.Div([
        header,
        rationale,
        html.Div(trace_items, style={"marginTop": "8px", "backgroundColor": "#fafafa",
                                      "padding": "8px 12px", "borderRadius": "4px",
                                      "border": "1px solid #e0e0e0"}),
    ], style={"padding": "10px 14px", "backgroundColor": "#f0f6ff",
              "borderLeft": "3px solid #06c", "borderRadius": "4px"})


# ─────────────────────────────────────────────────────────────────────────────
# Technicals tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_technicals(snapshots: list[TechnicalSnapshot]) -> html.Div:
    def _pct(v): return f"{v:+.2f}%" if v is not None else "—"
    def _num(v): return f"{v:.2f}" if v is not None else "—"

    def _rsi_color(rsi):
        if rsi is None:
            return "#888"
        if rsi >= 70:
            return "#b00"
        if rsi <= 30:
            return "#080"
        return "#333"

    def _rsi_int(v):
        return f"{int(round(v))}" if v is not None else "—"

    def _bb_cell_color(price, band_value, kind: str):
        if price is None or band_value is None:
            return "#333"
        if kind == "upper" and price >= band_value:
            return "#b00"
        if kind == "lower" and price <= band_value:
            return "#080"
        return "#333"

    def _rvol_color(rvol):
        if rvol is None:
            return "#888", "normal"
        if rvol > 2.0:
            return "#080", "bold"
        if rvol > 1.5:
            return "#080", "normal"
        if rvol < 0.5:
            return "#b00", "normal"
        return "#333", "normal"

    def _macd_signal_label(macd, signal, hist):
        if macd is None or signal is None or hist is None:
            return "—", "#888"
        if macd > signal and macd > 0:
            return "bullish ↑", "#080"
        if macd > signal and macd <= 0:
            return "recovering", "#5a5"
        if macd < signal and macd < 0:
            return "bearish ↓", "#b00"
        if macd < signal and macd >= 0:
            return "weakening", "#c60"
        return "neutral", "#888"

    rows = []
    for s in snapshots:
        if s.error:
            rows.append(html.Tr([
                html.Td(s.ticker, style={"fontWeight": "bold"}),
                html.Td(s.error, colSpan=13, style={"color": "red"}),
            ]))
            continue
        macd_label, macd_color = _macd_signal_label(s.macd, s.macd_signal, s.macd_hist)
        hist_color = "#080" if (s.macd_hist or 0) > 0 else ("#b00" if (s.macd_hist or 0) < 0 else "#333")
        c1 = "#080" if (s.change_1d_pct or 0) > 0 else ("#b00" if (s.change_1d_pct or 0) < 0 else "#333")
        rvol_color, rvol_weight = _rvol_color(s.rvol)
        rvol_text = f"{s.rvol:.2f}×" if s.rvol is not None else "—"

        rsi_bullish = s.rsi is not None and s.rsi > 50
        rvol_green = s.rvol is not None and s.rvol > 1.5
        macd_bullish = macd_label == "bullish ↑"
        if rsi_bullish and rvol_green and macd_bullish:
            action_text, action_color = "BUY", "#080"
        else:
            action_text, action_color = "—", "#888"

        rows.append(html.Tr([
            html.Td(f"{s.name} ({s.ticker})", style={"fontWeight": "bold"}),
            html.Td(f"${_num(s.price)}"),
            html.Td(_pct(s.change_1d_pct), style={"color": c1}),
            html.Td(_pct(s.change_5d_pct)),
            html.Td(_pct(s.change_20d_pct)),
            html.Td(_num(s.bb_upper), style={"color": _bb_cell_color(s.price, s.bb_upper, "upper")}),
            html.Td(_num(s.bb_middle)),
            html.Td(_num(s.bb_lower), style={"color": _bb_cell_color(s.price, s.bb_lower, "lower")}),
            html.Td(_rsi_int(s.rsi), style={"color": _rsi_color(s.rsi), "fontWeight": "bold"}),
            html.Td(_num(s.macd)),
            html.Td(_num(s.macd_hist), style={"color": hist_color, "fontWeight": "bold"}),
            html.Td(macd_label, style={"color": macd_color, "fontSize": "11px", "fontWeight": "bold"}),
            html.Td(rvol_text, style={"color": rvol_color, "fontWeight": rvol_weight}),
            html.Td(action_text, style={"color": action_color, "fontWeight": "bold",
                                         "fontSize": "12px", "textAlign": "center"}),
        ]))

    header = html.Tr([
        html.Th("Asset"), html.Th("Price"),
        html.Th("1d"), html.Th("5d"), html.Th("20d"),
        html.Th("上轨"), html.Th("中轨"), html.Th("下轨"),
        html.Th("RSI"),
        html.Th("MACD"), html.Th("Hist"), html.Th("Signal"),
        html.Th("RVOL"),
        html.Th("Action"),
    ])

    return html.Div([
        html.H4("Technicals — All Tracked Tickers"),
        html.Table([header] + rows,
                   style={"borderCollapse": "collapse", "width": "100%",
                          "fontSize": "12px", "textAlign": "left"}),
        html.P("BB: 20-day, 2σ | RSI: 14-day (≥70 overbought, ≤30 oversold) | "
               "MACD: 12/26/9 EMA — Hist = MACD − Signal (positive = bullish momentum accelerating) | "
               "RVOL: today's volume ÷ avg of prior 5 trading days (>1.5 elevated, >2.0 abnormal) | "
               "Action = BUY when RSI>50 AND RVOL>1.5 AND MACD bullish",
               style={"fontSize": "11px", "color": "#666", "marginTop": "8px"}),
    ], style={"overflowX": "auto"})


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_portfolio(result: dict) -> html.Div:
    if result.get("error") and not result.get("positions"):
        return html.P(result["error"], style={"color": "red"})

    fig = go.Figure()
    if result["dates"]:
        fig.add_trace(go.Scatter(
            x=result["dates"], y=result["total_values"],
            mode="lines", name="Total Portfolio Value",
            line={"color": "#0a6", "width": 2},
            fill="tozeroy", fillcolor="rgba(0,170,102,0.08)",
        ))
    fig.update_layout(
        title="Total Portfolio Value Over Time",
        xaxis_title="Date", yaxis_title="Value ($)",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        height=360, template="simple_white",
    )

    def _money(v): return f"${v:,.2f}" if v is not None else "—"
    def _pnl(v):
        if v is None:
            return html.Td("—")
        color = "#080" if v >= 0 else "#b00"
        return html.Td(f"{'+' if v >= 0 else ''}${v:,.2f}", style={"color": color, "fontWeight": "bold"})

    rows = []
    for p in result["positions"]:
        if p.error:
            rows.append(html.Tr([
                html.Td(p.ticker, style={"fontWeight": "bold"}),
                html.Td(f"{p.shares:g}"),
                html.Td(_money(p.invested)),
                html.Td(p.error, colSpan=3, style={"color": "red"}),
            ]))
            continue
        rows.append(html.Tr([
            html.Td(p.ticker, style={"fontWeight": "bold"}),
            html.Td(f"{p.shares:g}"),
            html.Td(f"${p.entry_price:,.2f}"),
            html.Td(_money(p.invested)),
            html.Td(_money(p.current_value)),
            _pnl(p.pnl),
        ]))

    header = html.Tr([
        html.Th("Ticker"), html.Th("Shares"), html.Th("Entry Price"),
        html.Th("Invested"), html.Th("Current Value"), html.Th("PnL"),
    ])

    current_total = result["total_values"][-1] if result["total_values"] else result["cash"]
    total_pnl = current_total - (result["invested"] + result["cash"])
    pnl_color = "#080" if total_pnl >= 0 else "#b00"
    summary = html.Div([
        html.Div([html.Span("Invested: ", style={"fontWeight": "bold"}),
                  html.Span(_money(result["invested"]))], style={"marginRight": "24px"}),
        html.Div([html.Span("Cash (remaining): ", style={"fontWeight": "bold"}),
                  html.Span(_money(result["cash"]))], style={"marginRight": "24px"}),
        html.Div([html.Span("Current Total: ", style={"fontWeight": "bold"}),
                  html.Span(_money(current_total))], style={"marginRight": "24px"}),
        html.Div([html.Span("Total PnL: ", style={"fontWeight": "bold"}),
                  html.Span(f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.2f}",
                           style={"color": pnl_color, "fontWeight": "bold"})]),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
              "padding": "12px", "backgroundColor": "#f8f8f8", "borderRadius": "6px",
              "marginBottom": "16px", "fontSize": "13px"})

    return html.Div([
        summary,
        dcc.Graph(figure=fig),
        html.H4("Per-Position Detail", style={"marginTop": "16px"}),
        html.Table([header] + rows,
                   style={"borderCollapse": "collapse", "width": "100%",
                          "fontSize": "13px", "textAlign": "left"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Fear & Greed tab
# ─────────────────────────────────────────────────────────────────────────────

def _fg_color(score: int) -> str:
    if score <= 24:
        return "#b00"
    if score <= 44:
        return "#e85"
    if score <= 55:
        return "#c90"
    if score <= 75:
        return "#6a0"
    return "#080"


def _classify(score: int) -> str:
    if score <= 24: return "Extreme Fear"
    if score <= 44: return "Fear"
    if score <= 55: return "Neutral"
    if score <= 75: return "Greed"
    return "Extreme Greed"


def _fg_gauge(idx: FearGreedIndex, title: str) -> dcc.Graph:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=idx.score,
        number={"font": {"size": 40, "color": _fg_color(idx.score)}},
        title={"text": f"<b>{title}</b><br><span style='font-size:14px;color:{_fg_color(idx.score)}'>"
                        f"{idx.classification}</span>",
                "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar": {"color": _fg_color(idx.score), "thickness": 0.25},
            "bgcolor": "#fff",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 25],  "color": "#fbb"},
                {"range": [25, 45], "color": "#fd9"},
                {"range": [45, 55], "color": "#ff9"},
                {"range": [55, 75], "color": "#cf8"},
                {"range": [75, 100], "color": "#8c8"},
            ],
            "threshold": {"line": {"color": "#222", "width": 3}, "thickness": 0.8,
                          "value": idx.score},
        },
    ))
    fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 80, "b": 20},
                      paper_bgcolor="#fff")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _render_fg_components(idx: FearGreedIndex) -> html.Div:
    rows = []
    for c in idx.components:
        color = _fg_color(c.score)
        badge = html.Span(_classify(c.score), style={
            "color": color, "fontWeight": "bold", "fontSize": "11px",
            "padding": "2px 8px", "borderRadius": "3px",
            "backgroundColor": f"{color}22", "border": f"1px solid {color}",
        })
        mock_tag = html.Span(" (mocked)", style={"fontSize": "10px", "color": "#999",
                                                 "fontStyle": "italic"}) if c.mocked else None
        rows.append(html.Tr([
            html.Td([c.name, mock_tag], style={"fontWeight": "bold"}),
            html.Td(c.detail, style={"fontSize": "11px", "color": "#555"}),
            html.Td(f"{c.score}", style={"color": color, "fontWeight": "bold",
                                          "textAlign": "right"}),
            html.Td(badge, style={"textAlign": "center"}),
        ]))

    header = html.Tr([
        html.Th("Component"), html.Th("Detail"),
        html.Th("Score", style={"textAlign": "right"}),
        html.Th("Rating", style={"textAlign": "center"}),
    ])
    return html.Table([header] + rows,
                      style={"borderCollapse": "collapse", "width": "100%",
                             "fontSize": "12px", "textAlign": "left"})


def _render_fg_history(idx: FearGreedIndex) -> html.Div:
    cells = []
    for label, score in idx.history.items():
        color = _fg_color(score)
        cells.append(html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#666"}),
            html.Div(str(score), style={"fontSize": "22px", "fontWeight": "bold",
                                         "color": color}),
            html.Div(_classify(score), style={"fontSize": "10px", "color": color}),
        ], style={"textAlign": "center", "padding": "8px 12px",
                  "border": "1px solid #ddd", "borderRadius": "6px",
                  "backgroundColor": "#fafafa", "flex": "1"}))
    return html.Div(cells, style={"display": "flex", "gap": "8px",
                                   "marginTop": "8px", "marginBottom": "16px"})


def _render_fg_panel(idx: FearGreedIndex, title: str) -> html.Div:
    if idx.error:
        return html.Div([
            html.H4(title),
            html.P(f"Error: {idx.error}", style={"color": "red"}),
        ])
    return html.Div([
        _fg_gauge(idx, title),
        html.Div("Historical", style={"fontSize": "12px", "fontWeight": "bold",
                                       "color": "#666", "marginTop": "-8px"}),
        _render_fg_history(idx),
        _render_fg_components(idx),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Macro tab (FRED + COT panels)
# ─────────────────────────────────────────────────────────────────────────────

SPARKLINE_SERIES = {"DGS10", "T10Y2Y", "DTWEXBGS", "VIXCLS", "BAMLH0A0HYM2"}


def _spark(series: FredSeries):
    if not series.observations:
        return html.Span("—", style={"color": "#999"})
    xs = [d for d, _ in series.observations[-60:]]
    ys = [v for _, v in series.observations[-60:]]
    color = "#080" if ys[-1] >= ys[0] else "#b00"
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": 1.5}))
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=40, width=140,
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                      style={"width": "140px", "height": "40px"})


def _render_macro_panel(panel: list[FredSeries]) -> html.Div:
    if all(s.error for s in panel):
        return html.Div([
            html.P("FRED data unavailable. Set FRED_API_KEY in .env.",
                    style={"color": "#b00"}),
        ])

    rows = []
    for s in panel:
        if s.error or not s.latest:
            rows.append(html.Tr([
                html.Td(f"{s.name} ({s.series_id})", style={"fontWeight": "bold"}),
                html.Td(s.error or "no data", colSpan=5,
                         style={"color": "#999", "fontStyle": "italic"}),
            ]))
            continue

        date, val = s.latest
        chg5 = s.change(5)
        chg20 = s.change(20)
        chg5_str = f"{chg5:+.2f}" if chg5 is not None else "—"
        chg20_str = f"{chg20:+.2f}" if chg20 is not None else "—"
        c5 = "#080" if (chg5 or 0) >= 0 else "#b00"
        c20 = "#080" if (chg20 or 0) >= 0 else "#b00"
        spark = _spark(s) if s.series_id in SPARKLINE_SERIES else html.Span("")

        rows.append(html.Tr([
            html.Td(f"{s.name} ({s.series_id})", style={"fontWeight": "bold"}),
            html.Td(f"{val:.2f} {s.units}".strip()),
            html.Td(date, style={"fontSize": "11px", "color": "#666"}),
            html.Td(chg5_str, style={"color": c5}),
            html.Td(chg20_str, style={"color": c20}),
            html.Td(spark),
        ]))

    header = html.Tr([
        html.Th("Series"), html.Th("Latest"), html.Th("As of"),
        html.Th("5d Δ"), html.Th("20d Δ"), html.Th("60d sparkline"),
    ])
    return html.Div([
        html.H4("Macro Backdrop"),
        html.Table([header] + rows,
                    style={"borderCollapse": "collapse", "width": "100%",
                           "fontSize": "13px", "textAlign": "left"}),
        html.P("Δ values are absolute (percentage points for rates/spreads, "
                "index points for DXY/VIX, $ for crude).",
                style={"fontSize": "11px", "color": "#888", "marginTop": "8px"}),
    ], style={"overflowX": "auto"})


def _render_cot_panel(panel: list[CotSnapshot]) -> html.Div:
    if all(s.error for s in panel):
        return html.Div([
            html.P("CFTC COT data unavailable.", style={"color": "#b00"}),
        ])

    rows = []
    for s in panel:
        if s.error or s.net_noncomm is None:
            rows.append(html.Tr([
                html.Td(f"{s.name} ({s.code})", style={"fontWeight": "bold"}),
                html.Td(s.error or "no data", colSpan=4,
                         style={"color": "#999", "fontStyle": "italic"}),
            ]))
            continue

        wc_str = f"{s.week_change:+,}" if s.week_change is not None else "—"
        wc_color = "#080" if (s.week_change or 0) >= 0 else "#b00"
        z_str = f"{s.z_score:+.2f}" if s.z_score is not None else "—"
        z_color = "#666"
        flag = ""
        if s.z_score is not None and s.z_score >= 2.0:
            z_color, flag = "#b00", "  crowded long"
        elif s.z_score is not None and s.z_score <= -2.0:
            z_color, flag = "#b00", "  crowded short"

        rows.append(html.Tr([
            html.Td(f"{s.name} ({s.code})", style={"fontWeight": "bold"}),
            html.Td(s.report_date, style={"fontSize": "11px", "color": "#666"}),
            html.Td(f"{s.net_noncomm:+,}"),
            html.Td(wc_str, style={"color": wc_color}),
            html.Td([z_str, html.Span(flag, style={"color": z_color, "fontSize": "11px"})],
                     style={"color": z_color, "fontWeight": "bold" if flag else "normal"}),
        ]))

    header = html.Tr([
        html.Th("Market"), html.Th("As of"), html.Th("Net spec"),
        html.Th("Δ wk"), html.Th("z (52w)"),
    ])
    return html.Div([
        html.H4("Speculator Positioning (CFTC COT)", style={"marginTop": "24px"}),
        html.Table([header] + rows,
                    style={"borderCollapse": "collapse", "width": "100%",
                           "fontSize": "13px", "textAlign": "left"}),
        html.P("Net spec = leveraged-fund (financials) or non-commercial (commodities) "
                "long − short. z-score vs trailing 52 weeks; |z|≥2 flagged as crowded.",
                style={"fontSize": "11px", "color": "#888", "marginTop": "8px"}),
    ], style={"overflowX": "auto"})


