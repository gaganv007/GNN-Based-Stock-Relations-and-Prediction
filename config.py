import os
from datetime import datetime

# --- Data parameters ---
START_DATE    = "2010-01-01"
SPLIT_DATE    = "2018-01-01"   # train=[2010–2017], test=[2018–today]
STOCKS        = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    "JPM","BAC","XOM","CVX","JNJ","PG","KO","WMT","HD"
]
# --- Graph parameters ---
CORRELATION_THRESHOLD = 0.3
SECTOR_MAPPING = {
    "Technology": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD"],
    "Finance":    ["JPM","BAC"],
    "Energy":     ["XOM","CVX"],
    "Healthcare": ["JNJ"],
    "Consumer":   ["PG","KO","WMT","HD"]
}
# --- Feature windows ---
TECHNICAL_INDICATORS = {
    "returns_1d": 1,
    "returns_5d": 5,
    "ma_10d":     10,
    "ma_20d":     20,
    "volatility_10d": 10,
    "rsi_14d":    14
}
# --- Model & training ---
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
LR            = 1e-3
EPOCHS        = 50
PATIENCE      = 5
BATCH_SIZE    = 16
