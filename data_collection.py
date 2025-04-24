import os
import pickle
import pandas as pd
import yfinance as yf
import config

class DataCollector:
    def download_stock_data(self, tickers=None, start_date=None, end_date=None):
        tickers    = tickers or config.STOCKS
        start_date = start_date or config.START_DATE
        end_date   = end_date   or config.END_DATE

        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            raw_dict = {t: df[t].copy() for t in df.columns.levels[0]}
        else:
            raw_dict = {tickers[0]: df.copy()}

        os.makedirs(config.RAW_DIR, exist_ok=True)
        out = os.path.join(config.RAW_DIR, "stock_data.pkl")
        with open(out, "wb") as f:
            pickle.dump(raw_dict, f)
        print(f"💾 Downloaded and cached stock data → {out}")
        return raw_dict

    def download_google_trends(self, *a, **k):
        return None

def download_all_data():
    dc = DataCollector()
    stock_dict = dc.download_stock_data()
    return stock_dict, None
