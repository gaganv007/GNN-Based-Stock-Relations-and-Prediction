import os
from datetime import datetime

# Data
START_DATE    = "2010-01-01"
SPLIT_DATE    = "2018-01-01"
END_DATE      = datetime.now().strftime("%Y-%m-%d")
DATA_DIR      = "data"
RAW_DIR       = os.path.join(DATA_DIR, "raw")

# Stocks to model (e.g. S&P 100 subset)
STOCKS = ["AAPL","MSFT","GOOGL","AMZN","FB","JPM","JNJ","V","PG","XOM"]

# Graph
CORR_THRESHOLD = 0.5

# GNN
DEVICE       = "cuda" if __import__("torch").cuda.is_available() else "cpu"
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
LR           = 1e-3
EPOCHS       = 50
BATCH_SIZE   = 1  # whole‐graph

# Results
MODELS_DIR   = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)
