# AI Macro Market Intelligence Agent

A Dash-based prototype that explains short-term asset moves by combining live web search, FinBERT sentiment, market data, and LLM-generated narratives — with built-in anti-hallucination validation.

## What It Does

Type any query (ticker, theme, or macro topic) and get:
- **Price snapshot** — current price, 1d/5d change
- **Headlines** — top news from Tavily, sorted by recency
- **Sentiment** — FinBERT or keyword-based scores
- **Narrative** — hedge-fund-style analysis via vLLM (template fallback)
- **Validation** — citation enforcement, numerical verification, confidence score

## Tracked Themes

| Theme | Coverage |
|---|---|
| **Oil** | WTI & Brent crude futures |
| **NeoCloud / AI Infra** | GPU cloud, hyperscalers, AI infrastructure plays |
| **Crypto** | Bitcoin, crypto miners, leveraged crypto equities |
| **AI Robotics** | Autonomous systems and robotics equities |
| **Credit** | HY/IG spreads, senior loans, EM debt, treasury benchmark |
| **Macro** | SPY, QQQ, TLT, DXY and cross-asset signals |
| **Custom Ticker** | Any yfinance-compatible symbol |

## Architecture

```
User query (Dash UI)
       │
       ▼
   Orchestrator (app.py)
   ├── Tavily Web Search     → top headlines
   ├── yfinance Market Data  → price, 1d%, 5d%
   ├── FinBERT Sentiment     → headline sentiment scores
   └── vLLM Narrative        → LLM analysis (fallback: template)
                └── Anti-Hallucination Validation
                      citation enforcement · numerical verification
                      uncertainty scoring · sentiment alignment
```

## UI Tabs

- **Analysis** — run a query and get narrative + validation panel
- **Technicals** — returns, Bollinger Bands, RSI across all tracked tickers
- **Portfolio Construction** — custom position sizing with P&L simulation

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add TAVILY_API_KEY
python app.py
```

Open http://localhost:8050.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TAVILY_API_KEY` | Yes (for live search) | API key from [tavily.com](https://tavily.com) |
| `VLLM_ENDPOINT` | No | vLLM server URL (e.g. `http://localhost:8000`) |
| `VLLM_API_KEY` | No | Auth key if vLLM server requires it |
| `VLLM_MODEL` | No | Model name (default: `meta-llama/Llama-3.3-70B-Instruct`) |
| `SENTIMENT_MODE` | No | `finbert` (default) or `mock` |

Both `TAVILY_API_KEY` and `VLLM_ENDPOINT` are optional — the app falls back to mock search and template narratives if either is absent.

## Self-Hosted LLM (vLLM)

Deploy on Nebius (~$5/hr, 2× A100) or AWS (p4d/p5):

```bash
pip install vllm
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 2
```

Set `VLLM_ENDPOINT=http://<host>:8000` in `.env`. For single-GPU, use `Llama-3.1-8B-Instruct`.

## Anti-Hallucination Validation

Every LLM response is validated for:
- **Citations** — every factual claim must reference a numbered source
- **Numerical accuracy** — extracted numbers cross-checked against market data
- **Confidence score** — 0–100, color-coded in the UI
- **Sentiment alignment** — narrative tone vs. FinBERT scores

```bash
python test_validation.py
```

## Project Structure

```
├── app.py                  # Dash UI + orchestrator
├── services/
│   ├── search.py           # Tavily web search
│   ├── market_data.py      # yfinance price retrieval
│   ├── sentiment.py        # FinBERT / mock sentiment
│   ├── narrative.py        # Template-based fallback narrative
│   ├── llm.py              # vLLM integration
│   └── validation.py       # Anti-hallucination validation
├── utils/
│   └── config.py           # Env vars + ticker constants
├── test_validation.py
├── requirements.txt
└── .env.example
```
