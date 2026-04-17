"""AI Macro Market Intelligence Agent — Dash frontend + orchestrator."""

from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, State, callback

from services.search import search_tavily, SearchResult
from services.market_data import get_snapshots_for_query, PriceSnapshot, get_credit_spreads, CreditSpread
from services.sentiment import analyze_sentiment, SentimentSummary
from services.narrative import generate_narrative
from services.llm import generate_and_validate_narrative
from services.validation import ValidationResult

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

app.layout = html.Div(
    style={"fontFamily": "Menlo, Consolas, monospace", "maxWidth": "960px",
           "margin": "0 auto", "padding": "24px"},
    children=[
        html.H2("Macro Market Intelligence Agent",
                 style={"borderBottom": "2px solid #333", "paddingBottom": "8px"}),

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


if __name__ == "__main__":
    app.run(debug=True, port=8050)
