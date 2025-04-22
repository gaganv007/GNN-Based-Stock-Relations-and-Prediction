import os
from datetime import datetime

# -----------------------------------------------------------------------------
# Data collection & train/test split
# -----------------------------------------------------------------------------
START_DATE      = "2010-01-01"
END_DATE        = datetime.now().strftime("%Y-%m-%d")

# Train on data ≤ this date; test on data ≥ TEST_START_DATE
TRAIN_END_DATE  = "2017-12-31"
TEST_START_DATE = "2018-01-01"

# -----------------------------------------------------------------------------
# API keys (set these as ENV vars or fill in directly)
# -----------------------------------------------------------------------------
FRED_API_KEY  = os.getenv("FRED_API_KEY",  "")  # Your FRED API key here
NEWS_API_KEY  = os.getenv("NEWS_API_KEY",  "")  # Your NewsAPI key here

# -----------------------------------------------------------------------------
# Equity universe
# -----------------------------------------------------------------------------
STOCKS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    # Finance
    "JPM", "BAC", "WFC",  "GS",   "V",    "MA",
    # Energy
    "XOM", "CVX", "COP",  "EOG",  "SLB",
    # Healthcare
    "JNJ", "UNH", "PFE",  "MRK",  "ABT",
    # Consumer
    "PG",  "KO",  "WMT",  "HD",   "COST"
]

# -----------------------------------------------------------------------------
# Cross‑asset features
# -----------------------------------------------------------------------------
# FRED series: CPI and Fed Funds Rate
MACRO_SERIES = {
    "CPIAUCSL": "CPI",
    "FEDFUNDS": "FedFunds"
}

# ETF tickers (use volume as proxy for flows)
ETF_TICKERS = ["SPY", "XLF", "XLE"]

# Commodity futures on Yahoo Finance
COMMODITY_TICKERS = ["GC=F", "CL=F"]

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
TECHNICAL_INDICATORS = {
    "returns_1d":    1,
    "returns_5d":    5,
    "ma_10d":       10,
    "volatility_5d": 5,
    # add RSI, momentum, etc. if you like
}

# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
CORRELATION_THRESHOLD = 0.3
SECTOR_MAPPING = {
    "Technology": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD"],
    "Finance":    ["JPM","BAC","WFC","GS","V","MA"],
    "Energy":     ["XOM","CVX","COP","EOG","SLB"],
    "Healthcare": ["JNJ","UNH","PFE","MRK","ABT"],
    "Consumer":   ["PG","KO","WMT","HD","COST"],
}

# -----------------------------------------------------------------------------
# GNN hyperparameters & training
# -----------------------------------------------------------------------------
# Input dim = #tech indic. + #macro series + #ETF vols + #commodities + 1 (sentiment)
INPUT_DIM  = (
    len(TECHNICAL_INDICATORS)
  + len(MACRO_SERIES)
  + len(ETF_TICKERS)
  + len(COMMODITY_TICKERS)
  + 1
)
HIDDEN_DIM = 64
OUTPUT_DIM = 2

LEARNING_RATE           = 0.001
NUM_EPOCHS              = 50
BATCH_SIZE              = 8
WEIGHT_DECAY            = 5e-4
EARLY_STOPPING_PATIENCE = 10

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR    = os.getcwd()
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
