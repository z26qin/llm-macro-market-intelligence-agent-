"""Macro Market Intelligence Agent — Dash entrypoint.

Thin wiring root. Layouts live in ui/layouts.py, renderers in ui/renderers.py,
callbacks in ui/callbacks.py, and the linear orchestrator in orchestrator.py.
The agent itself is in services/agent.py.
"""

from __future__ import annotations

import dash
from dash import DiskcacheManager
import diskcache

from ui.layouts import root_layout
import ui.callbacks  # noqa: F401 — importing registers the callbacks


_dash_cache = diskcache.Cache(".cache/dash")
_bg_manager = DiskcacheManager(_dash_cache)
app = dash.Dash(__name__, background_callback_manager=_bg_manager)
app.title = "Macro Market Intelligence"
app.layout = root_layout()


if __name__ == "__main__":
    app.run(debug=True, port=8051)
