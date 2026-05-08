"""Dash callbacks. Importing this module registers them as a side effect."""

from __future__ import annotations

import dash
from dash import callback, Input, Output, State, html

from orchestrator import run_analysis
from services.classifier import classify_query, ClassificationResult
from services.llm import is_llm_available
from services.agent import run_agent
from services.market_data import get_all_technical_snapshots
from services.fear_greed import get_cnn_fear_greed, get_crypto_fear_greed
from services.fred import fetch_default_panel as fetch_fred_panel
from services.cot import fetch_default_panel as fetch_cot_panel
from services.portfolio import compute_portfolio

from ui.renderers import (
    _render_prices, _render_credit_spreads, _render_sentiment,
    _render_headlines, _render_narrative, _render_validation,
    _render_classification, _render_debug,
    _render_agent_macro, _render_agent_positioning, _render_agent_options_iv,
    _render_live_trace, _render_agent_trace,
    _render_technicals, _render_portfolio,
    _render_fg_panel, _render_macro_panel, _render_cot_panel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tab — long callback with streaming progress
# ─────────────────────────────────────────────────────────────────────────────

def _run_agent_path(query: str, query_type_hint: str, set_progress=None) -> html.Div:
    hint = query_type_hint if query_type_hint != "auto" else "macro"
    if not is_llm_available():
        return html.Div([
            html.P("⚠ Agent mode requires a vLLM endpoint with tool-calling support.",
                   style={"color": "#b00", "fontWeight": "bold"}),
            html.P("Set VLLM_ENDPOINT and ensure the server was started with "
                   "`--enable-auto-tool-choice --tool-call-parser llama3_json`.",
                   style={"fontSize": "12px", "color": "#666"}),
        ])

    if set_progress is not None:
        set_progress(_render_live_trace([], stage="thinking"))

    def _on_step(entry, trace_so_far, stage):
        if set_progress is None:
            return
        set_progress(_render_live_trace(trace_so_far, stage))

    try:
        agent = run_agent(query, query_type_hint=hint, on_step=_on_step)
    except Exception as e:
        return html.Div([html.P(f"Agent failed: {e}", style={"color": "#b00"})])

    components = [_render_agent_trace(agent), html.Hr()]
    c = agent.collected
    if c.get("snapshots"):
        components.append(_render_prices(c["snapshots"])); components.append(html.Hr())
    if c.get("macro_panel") or c.get("macro"):
        components.append(_render_agent_macro(c)); components.append(html.Hr())
    if c.get("positioning_panel") or c.get("positioning"):
        components.append(_render_agent_positioning(c)); components.append(html.Hr())
    if c.get("options_iv"):
        components.append(_render_agent_options_iv(c["options_iv"])); components.append(html.Hr())
    if c.get("credit_spreads"):
        components.append(_render_credit_spreads(c["credit_spreads"])); components.append(html.Hr())
    if c.get("sentiment"):
        components.append(_render_sentiment(c["sentiment"])); components.append(html.Hr())
    if c.get("results"):
        components.append(_render_headlines(c["results"])); components.append(html.Hr())
    components.extend([
        _render_narrative(agent.narrative), html.Hr(),
        _render_validation(agent.validation),
    ])
    if agent.trace_path:
        components.append(html.P(f"Trace persisted: {agent.trace_path}",
                                  style={"fontSize": "11px", "color": "#888",
                                         "marginTop": "8px", "fontFamily": "Menlo, monospace"}))
    return html.Div(components)


@callback(
    Output("output-area", "children"),
    Input("run-btn", "n_clicks"),
    Input("linear-btn", "n_clicks"),
    State("query-input", "value"),
    State("query-type", "value"),
    background=True,
    progress=Output("agent-progress", "children"),
    prevent_initial_call=True,
)
def on_run(set_progress, run_clicks, linear_clicks, query, query_type):
    if not query:
        set_progress(html.Div())
        return html.P("Enter a query above.", style={"color": "#999"})

    trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else "run-btn"
    forced_linear = trigger == "linear-btn"

    if not forced_linear and is_llm_available():
        try:
            return _run_agent_path(query, query_type, set_progress=set_progress)
        except Exception as e:
            print(f"[app] agent path failed, falling back to linear: {e}")
            set_progress(html.Div(f"Agent path failed: {e} — falling back to linear.",
                                    style={"color": "#b00", "fontSize": "12px"}))

    set_progress(html.Div())

    classification: ClassificationResult | None = None
    if query_type == "auto":
        classification = classify_query(query)
        query_type = classification.query_type

    data = run_analysis(query, query_type)
    if "error" in data:
        return html.P(data["error"], style={"color": "red"})

    components = []
    if classification is not None:
        components.append(_render_classification(classification))
        components.append(html.Hr())
    components.extend([_render_prices(data["snapshots"]), html.Hr()])

    if data.get("query_type") == "credit" and data.get("credit_spreads"):
        components.append(_render_credit_spreads(data["credit_spreads"]))
        components.append(html.Hr())

    components.extend([
        _render_sentiment(data["sentiment"]), html.Hr(),
        _render_headlines(data["results"]), html.Hr(),
        _render_narrative(data["narrative"]), html.Hr(),
        _render_validation(data["validation"]), html.Hr(),
        _render_debug(data["results"], data["sentiment"]),
    ])
    return html.Div(components)


# ─────────────────────────────────────────────────────────────────────────────
# Technicals tab
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("tech-output", "children"),
    Input("tech-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_technicals(n_clicks):
    return _render_technicals(get_all_technical_snapshots())


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio tab
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Fear & Greed tab
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("fg-output", "children"),
    Input("fg-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_fear_greed(n_clicks):
    cnn = get_cnn_fear_greed()
    crypto = get_crypto_fear_greed()
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"},
        children=[
            _render_fg_panel(cnn, "CNN-Style Fear &amp; Greed"),
            _render_fg_panel(crypto, "Crypto Fear &amp; Greed"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Macro tab
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("macro-output", "children"),
    Input("macro-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_macro(n_clicks):
    return html.Div([
        _render_macro_panel(fetch_fred_panel()),
        _render_cot_panel(fetch_cot_panel()),
    ])
