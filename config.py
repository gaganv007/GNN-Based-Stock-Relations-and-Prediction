# config.py

import os
from datetime import datetime

BASE_DIR       = os.getcwd()
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
GRAPHS_DIR     = os.path.join(DATA_DIR, "graphs")
MODELS_DIR     = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

for d in (RAW_DIR, PROCESSED_DIR, GRAPHS_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Data collection window
START_DATE      = "2020-01-01"
END_DATE        = datetime.now().strftime("%Y-%m-%d")

# Split train/test
TEST_START_DATE = "2023-01-01"

# Feature engineering windows
FEATURE_WINDOW        = 10    # how many past days to flatten into features
DYNAMIC_CORR_WINDOW   = 20    # days for rolling correlation
SHORT_CORR_WINDOW     = 20
LONG_CORR_WINDOW      = 60

# Graph thresholds
CORRELATION_THRESHOLD = 0.4

# Model / training
MODELS     = ["GATWithAtt", "GCN", "GraphSAGE", "RandomForest"]
BATCH_SIZE = 16
LR         = 1e-3
EPOCHS     = 30

# Stocks list
STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    "JPM","BAC","WFC","GS","V","MA",
    "XOM","CVX","COP","EOG","SLB",
    "JNJ","UNH","PFE","MRK","ABT",
    "PG","KO","WMT","HD","COST"
]

# Technical indicators config
TECHNICAL_INDICATORS = {
    "returns_1d":    1,
    "returns_5d":    5,
    "ma_5d":         5,
    "ma_10d":       10,
    "ma_20d":       20,
    "volatility_5d": 5,
    "volatility_10d":10,
    "rsi_14d":      14,
    "momentum_5d":   5,
    "momentum_10d": 10,
}
