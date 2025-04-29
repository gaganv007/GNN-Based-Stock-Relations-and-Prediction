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

START_DATE      = "2010-01-01"
END_DATE        = datetime.now().strftime("%Y-%m-%d")
TEST_START_DATE = "2018-01-01"

FEATURE_WINDOW      = 10
TECHNICAL_INDICATORS = {
    "returns_1d":    1,   "returns_5d":    5,
    "ma_5d":         5,   "ma_10d":       10,  "ma_20d":       20,
    "volatility_5d": 5,   "volatility_10d":10,
    "rsi_14d":      14,   "momentum_5d":   5,  "momentum_10d": 10,
}
SMA_WINDOWS        = [9, 50, 100]
MFI_WINDOW         = 14

SHORT_CORR_WINDOW    = 20
LONG_CORR_WINDOW     = 60
CORRELATION_THRESHOLD= 0.4

SECTOR_MAP = {
    "Technology": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD"],
    "Finance":    ["JPM","BAC","WFC","GS","V","MA"],
    "Energy":     ["XOM","CVX","COP","EOG","SLB"],
    "Healthcare": ["JNJ","UNH","PFE","MRK","ABT"],
    "Consumer":   ["PG","KO","WMT","HD","COST"],
}

MODELS            = ["TemporalGAT","GATWithAtt","GCN","GraphSAGE","RandomForest"]
BATCH_SIZE        = 16
LR                = 1e-3
EPOCHS            = 30
DROP_EDGE_RATE    = 0.2
JITTER_STD        = 0.01
LABEL_SMOOTHING   = 0.1
LR_SCHEDULER      = "cyclic"
CYCLE_BASE_LR     = 1e-5
CYCLE_MAX_LR      = 1e-3
CYCLE_STEP_SIZE_UP= 200
GAT_HEADS         = 8
HID_DIM           = 128
NUM_LAYERS        = 3

STOCKS = sum(SECTOR_MAP.values(), [])