import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config

class DataCollector:
    def download_stock_data(self, tickers=None, start_date=None, end_date=None):
        tickers    = tickers or config.STOCKS
        start_date = start_date or config.START_DATE
        end_date   = end_date   or config.END_DATE

        print(f"Downloading stock data for {len(tickers)} stocks from {start_date} to {end_date}")
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        # for multi‐ticker, swap levels so data['AAPL']['Close'] works
        if len(tickers) > 1:
            data = data.swaplevel(0,1,axis=1).sort_index(axis=1)

        os.makedirs(config.RAW_DIR, exist_ok=True)
        raw_path = os.path.join(config.RAW_DIR, "stock_data.pkl")
        data.to_pickle(raw_path)
        print(f"💾 Downloaded and cached stock data → {raw_path}")
        return data

    def download_google_trends(self, tickers=None, lookback_days=None):
        # not implemented
        return None

def download_all_data():
    dc = DataCollector()
    stock_data  = dc.download_stock_data()
    trends_data = None
    if config.USE_GOOGLE_TRENDS:
        trends_data = dc.download_google_trends(
            tickers=config.STOCKS,
            lookback_days=config.GOOGLE_TRENDS_LOOKBACK
        )
    return stock_data, trends_data
