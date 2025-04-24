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

START_DATE      = "2020-01-01"
END_DATE        = datetime.now().strftime("%Y-%m-%d")
TEST_START_DATE = "2023-01-01"

STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    "JPM","BAC","WFC","GS","V","MA",
    "XOM","CVX","COP","EOG","SLB",
    "JNJ","UNH","PFE","MRK","ABT",
    "PG","KO","WMT","HD","COST"
]

TECHNICAL_INDICATORS = {
    "returns_1d":   1,
    "returns_5d":   5,
    "ma_5d":        5,
    "ma_10d":      10,
    "ma_20d":      20,
    "volatility_5d":5,
    "volatility_10d":10,
    "rsi_14d":     14,
    "momentum_5d":  5,
    "momentum_10d":10,
}

CORRELATION_THRESHOLD = 0.3
MODELS = ["GCN", "GAT", "GraphSAGE", "TemporalGNN", "RandomForest"]
