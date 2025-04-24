# config.py

import os
from datetime import datetime

# 1) Data dates
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"   # clamp to end of 2024

# 2) Stocks to include
STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    "JPM","BAC","WFC","GS","V","MA",
    "XOM","CVX","COP","EOG","SLB",
    "JNJ","UNH","PFE","MRK","ABT",
    "PG","KO","WMT","HD","COST"
]

# 3) Paths
BASE_DIR       = os.getcwd()
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
GRAPHS_DIR     = os.path.join(BASE_DIR, "data", "graphs")

# 4) GNN & training
MODELS         = ["GCN", "GAT", "GraphSAGE", "TemporalGNN"]
DEVICE         = "cuda" if False else "cpu"  # adjust if you have GPU
