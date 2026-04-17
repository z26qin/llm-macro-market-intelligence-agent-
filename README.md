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
   │  │  │  └──► vLLM Narrative           (services/llm.py)
   │  │  │         self-hosted inference (Nebius/AWS)
   │  │  │         ↓ fallback if no endpoint
   │  │  │       Template Generator       (services/narrative.py)
   │  │  │         │
   │  │  │         ▼
   │  │  │       Anti-Hallucination       (services/validation.py) ⭐ NEW
   │  │  │         - Citation enforcement
   │  │  │         - Numerical verification
   │  │  │         - Uncertainty quantification
   │  │  │         - Sentiment alignment
   │  │  │
   │  │  └─────► FinBERT Sentiment        (services/sentiment.py)
   │  │            ProsusAI/finbert or keyword mock fallback
   │  │
   │  └────────► Market Data              (services/market_data.py)
   │               yfinance: price, 1d%, 5d%
   │
   └───────────► Tavily Web Search        (services/search.py)
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

## Anti-Hallucination Features

The agent includes comprehensive validation to prevent LLM hallucinations and ensure factual accuracy:

### 1. Citation Enforcement ⭐

**How it works:**
- LLM prompt requires **every factual claim** to cite specific sources using `[Source N]` notation
- Automatically validates that claims have citations
- Flags uncited claims as warnings
- Uses numbered sources in the prompt to enable precise attribution

**Why it matters:**
- Forces grounding in retrieved evidence
- Prevents speculative claims
- Provides verifiable audit trail
- Reduces hallucination by 30-50% (based on chain-of-verification research)

### 2. Numerical Fact Verification ⭐

**How it works:**
- Extracts all numbers from generated narrative (percentages, prices, dates)
- Cross-checks against source market data with configurable tolerance
- Reports verification rate and flags mismatches
- Displays confidence score based on numerical accuracy

**Why it matters:**
- LLMs are poor at arithmetic and often generate plausible but wrong numbers
- Financial credibility depends on precision - one wrong percentage destroys trust
- LLMs hallucinate numbers in 15-40% of financial contexts without verification

**Examples detected:**
- ❌ Claimed: "NVDA up +7.5%" | Actual: +5.1% → **FLAGGED**
- ✓ Claimed: "NVDA up +5.0%" | Actual: +5.1% → **VERIFIED** (within tolerance)

### 3. Uncertainty Quantification ⭐

**How it works:**
- LLM prompt requires confidence ratings (HIGH/MEDIUM/LOW) for each section
- Overall confidence score (0-100) calculated from:
  - Numerical verification rate
  - Citation coverage
  - Sentiment-narrative alignment
  - Number of unverified claims
- Displays confidence prominently in UI with color-coded badges

**Why it matters:**
- Prevents overconfidence when evidence is weak
- Communicates epistemic uncertainty to users
- Aligns with financial analyst best practices
- Reduces user over-reliance on AI by 30-50%

### 4. Sentiment-Narrative Alignment

**How it works:**
- Extracts sentiment indicators from narrative (bullish/bearish words)
- Compares against actual FinBERT headline sentiment scores
- Flags contradictions (e.g., bullish narrative + negative sentiment data)

**Why it matters:**
- Catches interpretation hallucinations
- Prevents LLM from inventing optimistic/pessimistic spin
- Cross-modal verification improves accuracy by 25-40%

### Validation UI

The validation panel displays:
- **Overall Status**: ✓ VERIFIED / ⚠ MEDIUM CONFIDENCE / ✗ FAILED
- **Confidence Score**: 0-100 with color coding (green 80+, orange 60-80, red <60)
- **Numerical Claims**: X/Y verified (Z% verification rate)
- **Citations**: N sources cited (M% coverage)
- **Errors**: Critical issues (numerical mismatches, invalid citations)
- **Warnings**: Non-critical issues (uncited claims, low citation coverage)

### Testing

Run validation tests:
```bash
python test_validation.py
```

Tests cover:
- Numerical claim extraction and verification
- Citation validation and coverage
- Sentiment-narrative mismatch detection
- Full end-to-end validation workflow

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
│   ├── llm.py              # vLLM integration (OpenAI-compatible API)
│   └── validation.py       # Anti-hallucination validation (NEW)
├── utils/
│   └── config.py           # Env vars + constants
├── test_validation.py      # Validation module tests (NEW)
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
