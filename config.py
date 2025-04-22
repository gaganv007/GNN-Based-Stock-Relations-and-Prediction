import os
from datetime import datetime

# Date range
START_DATE = "2020-01-01"
END_DATE   = datetime.now().strftime("%Y-%m-%d")

# Stocks to include
STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA",
    "JPM","BAC","WFC","GS","V",
    "XOM","CVX","COP","EOG","SLB",
    "JNJ","UNH","PFE","MRK","ABT",
    "PG","KO","WMT","HD","COST"
]

# Graph parameters
CORRELATION_THRESHOLD = 0.3
SECTOR_MAPPING = {
    "Technology": ["AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA"],
    "Finance":    ["JPM","BAC","WFC","GS","V"],
    "Energy":     ["XOM","CVX","COP","EOG","SLB"],
    "Healthcare": ["JNJ","UNH","PFE","MRK","ABT"],
    "Consumer":   ["PG","KO","WMT","HD","COST"]
}

# Technical indicators
TECHNICAL_INDICATORS = {
    "returns_1d":    1,
    "returns_5d":    5,
    "ma_5d":         5,
    "ma_20d":       20,
    "volatility_5d": 5,
    "rsi_14d":      14
}

# GNN hyperparams
INPUT_DIM  = len(TECHNICAL_INDICATORS)
HIDDEN_DIM = 64
OUTPUT_DIM = 2

# Directories
BASE_DIR    = os.getcwd()
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
