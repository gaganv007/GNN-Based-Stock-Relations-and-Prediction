import os
import pandas as pd
import yfinance as yf
import config
from fredapi import Fred


class DataCollector:
    def __init__(self, fred_api_key=None):
        # pick up FRED_API_KEY from env if not passed
        self.fred = Fred(api_key=fred_api_key or os.getenv("FRED_API_KEY"))
        self.stock_data = None

    def download_stock_data(self, stocks=None, start_date=None, end_date=None):
        stocks     = stocks     or config.STOCKS
        start_date = start_date or config.START_DATE
        end_date   = end_date   or config.END_DATE

        print(f"   • Fetching {len(stocks)} tickers from {start_date} → {end_date}")
        df = yf.download(
            tickers    = stocks,
            start      = start_date,
            end        = end_date,
            group_by   = "ticker",
            auto_adjust= True,
            progress   = False
        )

        os.makedirs(os.path.join(config.DATA_DIR, "raw"), exist_ok=True)
        df.to_pickle(os.path.join(config.DATA_DIR, "raw", "stock_data.pkl"))
        self.stock_data = df
        return df

    # … keep any other methods (e.g. FRED series, news) …


# expose a top-level function for main.py
def download_price_data(stocks=None, start_date=None, end_date=None):
    dc = DataCollector()
    return dc.download_stock_data(stocks=stocks, start_date=start_date, end_date=end_date)


# backward-compat alias
download_all_data = download_price_data
