"""AI Macro Market Intelligence Agent — Dash frontend + orchestrator."""

from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go

from services.search import search_tavily, SearchResult
from services.market_data import (
    get_snapshots_for_query, PriceSnapshot,
    get_credit_spreads, CreditSpread,
    get_all_technical_snapshots, TechnicalSnapshot,
)
from services.sentiment import analyze_sentiment, SentimentSummary
from services.narrative import generate_narrative
from services.llm import generate_and_validate_narrative
from services.validation import ValidationResult
from services.portfolio import compute_portfolio, PositionResult

# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(query: str, query_type: str) -> dict:
    """Simple orchestrator: search -> market data -> sentiment -> narrative."""
    query = query.strip()
    if not query:
        return {"error": "Please enter a query."}

    # 1. Search
    results: list[SearchResult] = search_tavily(query, query_type)

    # 2. Market data
    snapshots: list[PriceSnapshot] = get_snapshots_for_query(query, query_type)

    # 3. Sentiment on headlines + snippets
    texts = []
    for r in results:
        if r.title:
            texts.append(r.title)
        if r.snippet:
            texts.append(r.snippet[:300])
    sentiment: SentimentSummary = analyze_sentiment(texts)

    # 4. Credit spreads (only for credit query type)
    credit_spreads: list[CreditSpread] = []
    if query_type == "credit":
        credit_spreads = get_credit_spreads()

    # 5. Narrative generation with validation (vLLM with template fallback)
    narrative, validation = generate_and_validate_narrative(
        topic=query,
        query_type=query_type,
        results=results,
        snapshots=snapshots,
        sentiment=sentiment,
        template_fallback_fn=generate_narrative,
    )

    return {
        "results": results,
        "snapshots": snapshots,
        "sentiment": sentiment,
        "narrative": narrative,
        "validation": validation,
        "credit_spreads": credit_spreads,
        "query_type": query_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dash App
# ═══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__)
app.title = "Macro Market Intelligence"

analysis_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
        # ── Input row ────────────────────────────────────────────────────
        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "flex-end",
                         "marginBottom": "16px"}, children=[
            html.Div(style={"flex": "1"}, children=[
                html.Label("Query", style={"fontSize": "13px", "fontWeight": "bold"}),
                dcc.Input(
                    id="query-input", type="text",
                    placeholder="e.g. NVDA, oil, AI infrastructure spending",
                    style={"width": "100%", "padding": "8px", "fontSize": "14px"},
                ),
            ]),
            html.Div(children=[
                html.Label("Type", style={"fontSize": "13px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id="query-type",
                    options=[
                        {"label": "Oil", "value": "oil"},
                        {"label": "NeoCloud / AI Infra", "value": "neocloud"},
                        {"label": "Crypto", "value": "crypto"},
                        {"label": "AI Robotics", "value": "ai_robotics"},
                        {"label": "Credit", "value": "credit"},
                        {"label": "Custom Ticker", "value": "ticker"},
                        {"label": "Macro Topic", "value": "macro"},
                    ],
                    value="ticker",
                    clearable=False,
                    style={"width": "200px", "fontSize": "14px"},
                ),
            ]),
            html.Button(
                "Run Analysis", id="run-btn",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
        ]),

        # ── Loading + output ─────────────────────────────────────────────
        dcc.Loading(id="loading", type="default", children=[
            html.Div(id="output-area", style={"marginTop": "12px"}),
        ]),
    ],
)


technicals_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                        "marginBottom": "16px"}, children=[
            html.Button(
                "Load Technicals", id="tech-btn",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
            html.Span("Returns, Bollinger Bands (20, 2σ), RSI (14) for all tracked tickers.",
                     style={"fontSize": "12px", "color": "#666"}),
        ]),
        dcc.Loading(id="tech-loading", type="default", children=[
            html.Div(id="tech-output", style={"marginTop": "12px"}),
        ]),
    ],
)


portfolio_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
        html.P("Positions are assumed entered on the first trading day of 2026 (2026-01-02). "
               "Cash earns no return.",
               style={"fontSize": "12px", "color": "#666", "marginBottom": "12px"}),

        html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                        "marginBottom": "12px"}, children=[
            html.Div(children=[
                html.Label("Total Capital ($)", style={"fontSize": "13px", "fontWeight": "bold"}),
                dcc.Input(
                    id="pf-capital", type="number", value=150000, min=0, step=1000,
                    style={"width": "160px", "padding": "8px", "fontSize": "14px"},
                ),
            ]),
            html.Button(
                "Calculate", id="pf-btn",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
        ]),

        html.Label("Positions", style={"fontSize": "13px", "fontWeight": "bold"}),
        dash_table.DataTable(
            id="pf-positions",
            columns=[
                {"name": "Ticker", "id": "ticker", "type": "text"},
                {"name": "Entry Price ($)", "id": "entry_price", "type": "numeric"},
                {"name": "Shares", "id": "shares", "type": "numeric"},
            ],
            data=[
                {"ticker": "MSTR", "entry_price": 141.0, "shares": 100},
                {"ticker": "MSTU", "entry_price": 7.0,   "shares": 1500},
                {"ticker": "TSLA", "entry_price": 382.0, "shares": 100},
                {"ticker": "TSLL", "entry_price": 13.0,  "shares": 2000},
                {"ticker": "CIFR", "entry_price": 17.6,  "shares": 300},
                {"ticker": "COHR", "entry_price": 308.0, "shares": 50},
                {"ticker": "STRC", "entry_price": 99.0,  "shares": 200},
                {"ticker": "IBIT", "entry_price": 50.0,  "shares": 100},
                {"ticker": "MRAL", "entry_price": 5.0,   "shares": 2200},
            ],
            editable=True,
            row_deletable=True,
            style_cell={"fontFamily": "Menlo, monospace", "fontSize": "13px", "padding": "6px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            style_table={"marginBottom": "12px"},
        ),
        html.Button(
            "+ Add Row", id="pf-add-row",
            style={"padding": "4px 12px", "fontSize": "12px", "cursor": "pointer",
                   "marginBottom": "16px", "border": "1px solid #888",
                   "backgroundColor": "#fff", "borderRadius": "4px"},
        ),

        dcc.Loading(id="pf-loading", type="default", children=[
            html.Div(id="pf-output", style={"marginTop": "12px"}),
        ]),
    ],
)


app.layout = html.Div(
    style={"fontFamily": "Menlo, Consolas, monospace", "maxWidth": "1152px",
           "margin": "0 auto", "padding": "24px"},
    children=[
        html.H2("Macro Market Intelligence Agent",
                 style={"borderBottom": "2px solid #333", "paddingBottom": "8px"}),
        dcc.Tabs(id="main-tabs", value="analysis", children=[
            dcc.Tab(label="Analysis", value="analysis", children=[analysis_tab]),
            dcc.Tab(label="Technicals", value="technicals", children=[technicals_tab]),
            dcc.Tab(label="Portfolio Construction", value="portfolio", children=[portfolio_tab]),
        ]),
    ],
)


# ── Helper renderers ─────────────────────────────────────────────────────────

def _format_date(date_str: str | None) -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return date_str[:10] if date_str else ""


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
    """Render credit spread table."""
    if not spreads:
        return html.Div()

    rows = []
    for sp in spreads:
        d1 = f"{sp.spread_1d_change:+.3f}%" if sp.spread_1d_change is not None else "—"
        d5 = f"{sp.spread_5d_change:+.3f}%" if sp.spread_5d_change is not None else "—"
        # Color based on interpretation
        if "tightening" in sp.interpretation or "outperforming" in sp.interpretation or "risk-on" in sp.interpretation:
            color = "#080"  # green = risk-on
        elif "widening" in sp.interpretation or "underperforming" in sp.interpretation or "risk-off" in sp.interpretation or "stress" in sp.interpretation:
            color = "#b00"  # red = risk-off
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
        html.Th("Spread"),
        html.Th("Value"),
        html.Th("1d Δ"),
        html.Th("5d Δ"),
        html.Th("Interpretation"),
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
    """Render validation results with confidence score and warnings."""
    # Determine overall status color
    if validation.passed and validation.confidence_score >= 80:
        status_color = "#080"  # green
        status_text = "✓ VERIFIED"
    elif validation.passed and validation.confidence_score >= 60:
        status_color = "#f90"  # orange
        status_text = "⚠ VERIFIED (Medium Confidence)"
    elif validation.passed:
        status_color = "#f90"  # orange
        status_text = "⚠ VERIFIED (Low Confidence)"
    else:
        status_color = "#b00"  # red
        status_text = "✗ VALIDATION FAILED"

    # Build validation details
    details_items = []

    # Confidence score
    details_items.append(
        html.Div([
            html.Span("Overall Confidence: ", style={"fontWeight": "bold"}),
            html.Span(f"{validation.confidence_score:.0f}/100",
                     style={"color": status_color, "fontWeight": "bold", "fontSize": "16px"}),
        ], style={"marginBottom": "8px"})
    )

    # Numerical verification stats
    num_ver = validation.numerical_verification
    details_items.append(
        html.Div([
            html.Span("Numerical Claims: ", style={"fontWeight": "bold"}),
            html.Span(f"{num_ver['verified_claims']}/{num_ver['total_claims']} verified"),
            html.Span(f" ({num_ver['verification_rate']*100:.0f}%)",
                     style={"fontSize": "11px", "color": "#666"}),
        ], style={"marginBottom": "4px"})
    )

    # Citation verification stats
    cite_ver = validation.citation_verification
    details_items.append(
        html.Div([
            html.Span("Citations: ", style={"fontWeight": "bold"}),
            html.Span(f"{cite_ver['unique_citations']} sources cited"),
            html.Span(f" ({cite_ver['citation_coverage']*100:.0f}% coverage)",
                     style={"fontSize": "11px", "color": "#666"}),
        ], style={"marginBottom": "4px"})
    )

    # Errors
    if validation.errors:
        error_list = html.Ul([html.Li(err, style={"color": "#b00"}) for err in validation.errors],
                            style={"marginLeft": "16px", "fontSize": "12px"})
        details_items.append(
            html.Div([
                html.Span("❌ Errors:", style={"fontWeight": "bold", "color": "#b00"}),
                error_list,
            ], style={"marginTop": "8px"})
        )

    # Warnings
    if validation.warnings:
        warning_list = html.Ul([html.Li(warn, style={"color": "#f90"}) for warn in validation.warnings],
                              style={"marginLeft": "16px", "fontSize": "12px"})
        details_items.append(
            html.Div([
                html.Span("⚠️ Warnings:", style={"fontWeight": "bold", "color": "#f90"}),
                warning_list,
            ], style={"marginTop": "8px"})
        )

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


# ── Callback ─────────────────────────────────────────────────────────────────

@callback(
    Output("output-area", "children"),
    Input("run-btn", "n_clicks"),
    State("query-input", "value"),
    State("query-type", "value"),
    prevent_initial_call=True,
)
def on_run(n_clicks, query, query_type):
    if not query:
        return html.P("Enter a query above.", style={"color": "#999"})

    data = run_analysis(query, query_type)

    if "error" in data:
        return html.P(data["error"], style={"color": "red"})

    # Build output components
    components = [
        _render_prices(data["snapshots"]),
        html.Hr(),
    ]

    # Add credit spreads section if this is a credit query
    if data.get("query_type") == "credit" and data.get("credit_spreads"):
        components.append(_render_credit_spreads(data["credit_spreads"]))
        components.append(html.Hr())

    components.extend([
        _render_sentiment(data["sentiment"]),
        html.Hr(),
        _render_headlines(data["results"]),
        html.Hr(),
        _render_narrative(data["narrative"]),
        html.Hr(),
        _render_validation(data["validation"]),
        html.Hr(),
        _render_debug(data["results"], data["sentiment"]),
    ])

    return html.Div(components)


def _render_technicals(snapshots: list[TechnicalSnapshot]) -> html.Div:
    """Render technicals table: returns + Bollinger Bands + RSI."""
    def _pct(v): return f"{v:+.2f}%" if v is not None else "—"
    def _num(v): return f"{v:.2f}" if v is not None else "—"

    def _rsi_color(rsi):
        if rsi is None:
            return "#888"
        if rsi >= 70:
            return "#b00"  # overbought
        if rsi <= 30:
            return "#080"  # oversold
        return "#333"

    def _rsi_int(v):
        return f"{int(round(v))}" if v is not None else "—"

    def _bb_cell_color(price, band_value, kind: str):
        """Color upper red when price pierces above, lower green when price pierces below."""
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
            return "#080", "bold"   # abnormal volume — strong green
        if rvol > 1.5:
            return "#080", "normal"  # elevated — green
        if rvol < 0.5:
            return "#b00", "normal"  # unusually light
        return "#333", "normal"

    def _macd_signal_label(macd, signal, hist):
        """Interpret MACD state for a quick read."""
        if macd is None or signal is None or hist is None:
            return "—", "#888"
        # Bullish: MACD > signal (above zero = strong bullish momentum)
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

        # Action: Buy if RSI > 50 (bullish bias) AND RVOL elevated (>1.5, green) AND MACD bullish
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


@callback(
    Output("tech-output", "children"),
    Input("tech-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_technicals(n_clicks):
    snapshots = get_all_technical_snapshots()
    return _render_technicals(snapshots)


def _render_portfolio(result: dict) -> html.Div:
    """Render portfolio chart + per-position table + summary."""
    if result.get("error") and not result.get("positions"):
        return html.P(result["error"], style={"color": "red"})

    # Time-series chart
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

    # Per-position table
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

    # Summary
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


@callback(
    Output("pf-positions", "data"),
    Input("pf-add-row", "n_clicks"),
    State("pf-positions", "data"),
    prevent_initial_call=True,
)
def on_add_portfolio_row(n_clicks, rows):
    rows = list(rows or [])
    rows.append({"ticker": "", "entry_price": None, "shares": None})
    return rows


@callback(
    Output("pf-output", "children"),
    Input("pf-btn", "n_clicks"),
    State("pf-positions", "data"),
    State("pf-capital", "value"),
    prevent_initial_call=True,
)
def on_compute_portfolio(n_clicks, positions, capital):
    if not capital or capital <= 0:
        return html.P("Enter a positive Total Capital.", style={"color": "red"})
    result = compute_portfolio(positions or [], float(capital))
    return _render_portfolio(result)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
