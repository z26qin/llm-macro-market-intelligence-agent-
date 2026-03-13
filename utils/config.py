"""Configuration and constants for the macro market intelligence agent."""

import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Query type constants
QUERY_OIL = "oil"
QUERY_NEOCLOUD = "neocloud"
QUERY_TICKER = "ticker"
QUERY_MACRO = "macro"

# Ticker mappings for common query types
OIL_TICKERS = ["CL=F", "BZ=F"]
OIL_DISPLAY = {"CL=F": "WTI Crude", "BZ=F": "Brent Crude"}

NEOCLOUD_TICKERS = ["NVDA", "NBIS", "CRWV", "ORCL", "MSFT", "IREN"]

CRYPTO_TICKERS = ["BMNR", "MSTR", "BTC-USD"]

AI_ROBOTICS_TICKERS = ["TSLA"]

CREDIT_TICKERS = [
    "HYG",   # High Yield Corporate Bond
    "LQD",   # Investment Grade Corporate Bond
    "JNK",   # SPDR High Yield Bond
    "BKLN",  # Senior Loans (liquidity indicator)
    "EMB",   # EM USD Bond
    "TLT",   # Treasury benchmark (for spread calc)
]

# Tavily search defaults
SEARCH_MAX_RESULTS = 8

# Sentiment mode: "finbert" or "mock"
SENTIMENT_MODE: str = os.getenv("SENTIMENT_MODE", "finbert")
