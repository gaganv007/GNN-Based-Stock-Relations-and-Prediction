import os
import yfinance as yf
import pandas as pd
import config

def download_all_data():
    """
    Download historical price data into data/raw and
    save per‐ticker DataFrames in data/processed.
    """
    raw = yf.download(
        tickers=config.STOCKS,
        start=config.START_DATE,
        end=config.END_DATE,
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw.to_pickle(os.path.join(raw_dir, "stock_data.pkl"))

    # Unpack into dict of per‐ticker DataFrames
    processed = {}
    for stk in config.STOCKS:
        processed[stk] = raw[stk].copy()

    proc_dir = os.path.join(config.DATA_DIR, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    pd.to_pickle(processed, os.path.join(proc_dir, "merged_data.pkl"))

    return processed

if __name__ == "__main__":
    download_all_data()
