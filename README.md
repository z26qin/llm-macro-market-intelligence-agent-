# AI Macro Market Intelligence Agent

A lightweight prototype that explains short-term moves in assets (oil, AI infra / neocloud stocks, major tech names) by combining web search, sentiment analysis, market data, and narrative generation.

## Architecture

```
User input (Dash UI)
       │
       ▼
┌─────────────────────┐
│   Orchestrator       │  app.py :: run_analysis()
│   (query classifier) │
└──┬──┬──┬──┬─────────┘
   │  │  │  │
   │  │  │  └──► Nebius LLM Narrative  (services/llm_nebius.py)
   │  │  │         cloud inference via Token Factory
   │  │  │         ↓ fallback if no API key
   │  │  │       Template Generator    (services/narrative.py)
   │  │  │
   │  │  └─────► FinBERT Sentiment     (services/sentiment.py)
   │  │            ProsusAI/finbert or keyword mock fallback
   │  │
   │  └────────► Market Data           (services/market_data.py)
   │               yfinance: price, 1d%, 5d%
   │
   └───────────► Tavily Web Search     (services/search.py)
                   top headlines + snippets
```

## Setup

```bash
# 1. Clone and enter directory
cd llm-macro-market-intelligence-agent

# 2. Create virtualenv
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY

# 5. Run
python app.py
```

Open http://localhost:8050 in your browser.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TAVILY_API_KEY` | Yes (for live search) | API key from [tavily.com](https://tavily.com) |
| `NEBIUS_API_KEY` | No | API key for Nebius Token Factory LLM inference |
| `SENTIMENT_MODE` | No | `finbert` (default) or `mock` for lightweight keyword heuristic |

If `TAVILY_API_KEY` is not set, the app returns mock search results and still runs.

If FinBERT fails to load (e.g. no torch), the app falls back to mock sentiment automatically.

## Using Nebius Token Factory

The app supports cloud LLM inference via [Nebius Token Factory](https://studio.nebius.com/) for generating rich, hedge-fund-style market narratives.

### Setup

1. Get an API key from [Nebius AI Studio](https://studio.nebius.com/)
2. Add it to your `.env` file:
   ```
   NEBIUS_API_KEY=your_api_key_here
   ```

### How It Works

When `NEBIUS_API_KEY` is set, the narrative generator uses Nebius Token Factory to call a Llama 3.3 70B model. The model receives:
- Current price data (ticker, price, 1d/5d changes)
- Top news headlines from Tavily search
- Sentiment summary from FinBERT analysis

The LLM produces a structured analysis with:
1. Move Summary
2. Likely Drivers
3. Market Interpretation
4. Confidence Level

### Fallback Mode

If `NEBIUS_API_KEY` is not set or if the API call fails, the app automatically falls back to the built-in template-based narrative generator. This ensures the app always works, even without cloud inference.

## Example Queries

| Query | Type | What it does |
|---|---|---|
| `oil` | Oil | WTI + Brent prices, oil-related headlines, sentiment |
| `NVDA` | Custom Ticker | NVDA price snapshot, stock-specific news, narrative |
| `NBIS` | NeoCloud / AI Infra | All neocloud tickers + AI infra news |
| `tariffs` | Macro Topic | SPY/QQQ/TLT/DXY + macro headlines |
| `CRWV` | Custom Ticker | CoreWeave specific analysis |

## Project Structure

```
├── app.py                  # Dash UI + orchestrator
├── services/
│   ├── search.py           # Tavily web search
│   ├── market_data.py      # yfinance price retrieval
│   ├── sentiment.py        # FinBERT / mock sentiment
│   ├── narrative.py        # Template-based narrative generator (fallback)
│   └── llm_nebius.py       # Nebius Token Factory LLM integration
├── utils/
│   └── config.py           # Env vars + constants
├── requirements.txt
├── .env.example
└── README.md
```

## Future Improvements

- **LangGraph agent** — replace the simple orchestrator with a stateful agent graph
- **Redis caching** — cache search results and price data to reduce API calls
- **Vector store / RAG memory** — store past analyses for cross-session context
- **Streaming UI** — server-sent events for real-time narrative generation
- **Multi-asset correlation view** — show how moves relate across asset classes
- **Event timeline** — chronological view of catalysts driving price action
- **Better sentiment calibration** — fine-tune FinBERT on financial headlines, add entity-level sentiment
- **LLM narrative upgrade** — swap template engine for Claude / GPT API for richer commentary
- **Alerting** — scheduled runs with threshold-based notifications
