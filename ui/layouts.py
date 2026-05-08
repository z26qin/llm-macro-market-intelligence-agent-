"""Tab layouts for the Dash app. Pure presentation — no callbacks."""

from __future__ import annotations

from dash import dcc, html, dash_table


analysis_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
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
                        {"label": "Auto (classify)", "value": "auto"},
                        {"label": "Oil", "value": "oil"},
                        {"label": "NeoCloud / AI Infra", "value": "neocloud"},
                        {"label": "Crypto", "value": "crypto"},
                        {"label": "AI Robotics", "value": "ai_robotics"},
                        {"label": "Credit", "value": "credit"},
                        {"label": "Custom Ticker", "value": "ticker"},
                        {"label": "Macro Topic", "value": "macro"},
                    ],
                    value="auto",
                    clearable=False,
                    style={"width": "200px", "fontSize": "14px"},
                ),
            ]),
            html.Button(
                "Run Analysis", id="run-btn",
                title="Agent path when vLLM is available; falls back to the linear orchestrator otherwise.",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
            html.Button(
                "Force Linear", id="linear-btn",
                title="Skip the agent and use the fixed pipeline (debugging).",
                style={"padding": "8px 16px", "fontSize": "12px",
                       "cursor": "pointer", "backgroundColor": "#fff",
                       "color": "#222", "border": "1px solid #888", "borderRadius": "4px"},
            ),
        ]),
        html.Div(id="agent-progress", style={"marginTop": "12px"}),
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


fear_greed_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                        "marginBottom": "16px"}, children=[
            html.Button(
                "Refresh Indices", id="fg-btn",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
            html.Span("CNN-style Fear & Greed + Crypto Fear & Greed. "
                      "Live components computed from SPY/VIX/TLT/HYG/BTC; "
                      "items marked (mocked) use seeded values stable within a day.",
                      style={"fontSize": "12px", "color": "#666"}),
        ]),
        dcc.Loading(id="fg-loading", type="default", children=[
            html.Div(id="fg-output", style={"marginTop": "12px"}),
        ]),
    ],
)


macro_tab = html.Div(
    style={"paddingTop": "16px"},
    children=[
        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                        "marginBottom": "16px"}, children=[
            html.Button(
                "Refresh Macro", id="macro-btn",
                style={"padding": "8px 20px", "fontSize": "14px",
                       "cursor": "pointer", "backgroundColor": "#222",
                       "color": "#fff", "border": "none", "borderRadius": "4px"},
            ),
            html.Span("Rates, dollar, vol, credit spreads, breakevens — sourced from FRED. "
                      "Cached on disk for 6 hours.",
                      style={"fontSize": "12px", "color": "#666"}),
        ]),
        dcc.Loading(id="macro-loading", type="default", children=[
            html.Div(id="macro-output", style={"marginTop": "12px"}),
        ]),
    ],
)


def root_layout() -> html.Div:
    return html.Div(
        style={"fontFamily": "Menlo, Consolas, monospace", "maxWidth": "1152px",
               "margin": "0 auto", "padding": "24px"},
        children=[
            html.H2("Macro Market Intelligence Agent",
                   style={"borderBottom": "2px solid #333", "paddingBottom": "8px"}),
            dcc.Tabs(id="main-tabs", value="analysis", children=[
                dcc.Tab(label="Analysis", value="analysis", children=[analysis_tab]),
                dcc.Tab(label="Technicals", value="technicals", children=[technicals_tab]),
                dcc.Tab(label="Macro", value="macro", children=[macro_tab]),
                dcc.Tab(label="Portfolio Construction", value="portfolio", children=[portfolio_tab]),
                dcc.Tab(label="Fear & Greed", value="fear_greed", children=[fear_greed_tab]),
            ]),
        ],
    )
