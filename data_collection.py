import yfinance as yf
import pandas as pd
from tqdm import tqdm
from config import STOCKS, START_DATE
import os

def download_price_data():
    """
    Download adjusted close prices for all STOCKS since START_DATE.
    Returns a DataFrame with date index and tickers as columns.
    """
    os.makedirs("data", exist_ok=True)
    cache_file = "data/price_data.pkl"
    try:
        df = pd.read_pickle(cache_file)
        print("✅ Loaded cached price data")
        return df
    except FileNotFoundError:
        pass

    print("↓ Downloading price data …")
    data = yf.download(
        tickers=STOCKS,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )["Close"]
    data = data.dropna(how="all")  # drop days with no market data
    data.to_pickle(cache_file)
    print("✅ Saved price data to cache")
    return data
