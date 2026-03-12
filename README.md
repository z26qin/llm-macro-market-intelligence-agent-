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
   │  │  │  └──► vLLM Narrative        (services/llm.py)
   │  │  │         self-hosted inference (Nebius/AWS)
   │  │  │         ↓ fallback if no endpoint
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
| `VLLM_ENDPOINT` | No | vLLM server URL (e.g., `http://localhost:8000`) |
| `VLLM_API_KEY` | No | Optional API key if vLLM server requires auth |
| `VLLM_MODEL` | No | Model name (default: `meta-llama/Llama-3.3-70B-Instruct`) |
| `SENTIMENT_MODE` | No | `finbert` (default) or `mock` for lightweight keyword heuristic |

If `TAVILY_API_KEY` is not set, the app returns mock search results and still runs.

If FinBERT fails to load (e.g. no torch), the app falls back to mock sentiment automatically.

## Using vLLM for Narrative Generation

The app supports self-hosted LLM inference via [vLLM](https://github.com/vllm-project/vllm) for generating hedge-fund-style market narratives.

### How It Works

When `VLLM_ENDPOINT` is set, the narrative generator calls your vLLM server with:
- Current price data (ticker, price, 1d/5d changes)
- Top news headlines from Tavily search
- Sentiment summary from FinBERT analysis

The LLM produces a structured analysis with:
1. Move Summary
2. Likely Drivers
3. Market Interpretation
4. Confidence Level

### Fallback Mode

If `VLLM_ENDPOINT` is not set or if the API call fails, the app automatically falls back to the built-in template-based narrative generator.

### Deploying vLLM on Nebius Cloud

Nebius offers cost-effective GPU VMs with good H100/A100 availability.

```bash
# 1. Create a Nebius VM with GPU (e.g., 2x A100 80GB for 70B model)

# 2. SSH into the VM and install vLLM
pip install vllm

# 3. Start the server
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2

# 4. Configure your .env
VLLM_ENDPOINT=http://<nebius-vm-ip>:8000
```

### Deploying vLLM on AWS

AWS offers p4d (A100) and p5 (H100) instances.

```bash
# 1. Launch an EC2 instance (e.g., p4d.24xlarge with 8x A100)

# 2. SSH into the instance and install vLLM
pip install vllm

# 3. Start the server
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2

# 4. Configure your .env (use private IP if in same VPC)
VLLM_ENDPOINT=http://<ec2-ip>:8000
```

### Using a Smaller Model

For lower GPU requirements, use a smaller model:

```bash
# 8B model - fits on single A100/H100
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

# Update .env
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Cost Comparison

| Provider | GPU | Approx. Cost |
|----------|-----|--------------|
| Nebius | 2x A100 80GB | ~$5/hr |
| AWS | p4d.24xlarge (8x A100) | ~$32/hr |
| AWS | p5.48xlarge (8x H100) | ~$98/hr |

For a 70B model, 2x A100 80GB is sufficient. Nebius is more cost-effective for dedicated GPU workloads.

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
│   └── llm.py              # vLLM integration (OpenAI-compatible API)
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
- **Alerting** — scheduled runs with threshold-based notifications
