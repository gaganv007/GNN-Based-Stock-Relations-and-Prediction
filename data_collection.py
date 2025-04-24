# data_collection.py
import os, pandas as pd, yfinance as yf
from tqdm import tqdm
import config

class DataCollector:
    def __init__(self):
        os.makedirs(config.RAW_DIR, exist_ok=True)

    def download_price_data(self, force=False):
        for tkr in tqdm(config.STOCKS, desc="Downloading"):
            path = os.path.join(config.RAW_DIR, f"{tkr}.pkl")
            if os.path.exists(path) and not force:
                continue
            df = yf.download(tkr, start=config.START_DATE, end=config.END_DATE,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            df["return_1d"] = df["Close"].pct_change().shift(-1)  # next‐day
            df.to_pickle(path)
        print("✅ Prices cached.")

    def load_all(self):
        data = {}
        for tkr in config.STOCKS:
            path = os.path.join(config.RAW_DIR, f"{tkr}.pkl")
            if os.path.exists(path):
                data[tkr] = pd.read_pickle(path)
        return data
