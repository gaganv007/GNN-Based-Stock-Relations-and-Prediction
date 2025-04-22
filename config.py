import os
from datetime import datetime

START_DATE = "2020-01-01"
END_DATE   = datetime.now().strftime("%Y-%m-%d")

STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD",
    "JPM","BAC","WFC","GS","V","MA",
    "XOM","CVX","COP","EOG","SLB",
    "JNJ","UNH","PFE","MRK","ABT",
    "PG","KO","WMT","HD","COST"
]

CORRELATION_THRESHOLD    = 0.3
SECTOR_MAPPING = {
    "Technology":[ "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD" ],
    "Finance":   [ "JPM","BAC","WFC","GS","V","MA" ],
    "Energy":    [ "XOM","CVX","COP","EOG","SLB" ],
    "Healthcare":[ "JNJ","UNH","PFE","MRK","ABT" ],
    "Consumer":  [ "PG","KO","WMT","HD","COST" ]
}

USE_GOOGLE_TRENDS      = True
GOOGLE_TRENDS_LOOKBACK = 30
GOOGLE_TRENDS_BATCH_SIZE = 5
GOOGLE_TRENDS_MAX_RETRIES = 3
GOOGLE_TRENDS_RETRY_DELAY  = 1  # sec

TECHNICAL_INDICATORS = {
    "returns_1d":1, "returns_5d":5,
    "ma_5d":5,     "ma_10d":10,  "ma_20d":20,
    "volatility_5d":5, "volatility_10d":10,
    "rsi_14d":14,
    "momentum_5d":5,   "momentum_10d":10
}

MODELS      = ["GCN","GAT","GraphSAGE","TemporalGNN","LSTM","RandomForest"]
INPUT_DIM   = len(TECHNICAL_INDICATORS) + (1 if USE_GOOGLE_TRENDS else 0)
HIDDEN_DIM  = 64
OUTPUT_DIM  = 2

BATCH_SIZE    = 32
LEARNING_RATE = 0.001
NUM_EPOCHS    = 50
PATIENCE      = 10
WEIGHT_DECAY  = 1e-5

DATA_DIR     = os.path.join(os.getcwd(),"data")
MODELS_DIR   = os.path.join(os.getcwd(),"saved_models")
RESULTS_DIR  = os.path.join(os.getcwd(),"results")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
