# data_collection.py

import os
import pandas as pd
import yfinance as yf
import config

class DataCollector:
    """
    Download & cache raw price data to CSV.
    """
    def __init__(self):
        os.makedirs(config.RAW_DIR, exist_ok=True)
        self.csv_path = os.path.join(config.RAW_DIR, "stock_data.csv")

    def load_or_download(self):
        if os.path.exists(self.csv_path):
            print("✅ Raw CSV cache found.")
            df = pd.read_csv(self.csv_path, parse_dates=["Date"], index_col="Date")
        else:
            print(f"🔄 Downloading {len(config.STOCKS)} tickers from {config.START_DATE} to {config.END_DATE}…")
            df_list = []
            for t in config.STOCKS:
                tmp = (
                    yf.download(t,
                                start=config.START_DATE,
                                end=config.END_DATE,
                                progress=False)
                      .assign(Ticker=t)
                      .reset_index()[["Date","Ticker","Open","High","Low","Close","Volume"]]
                )
                df_list.append(tmp)
            df = pd.concat(df_list, axis=0)
            df.to_csv(self.csv_path, index=False)
            print(f"💾 Raw prices saved to {self.csv_path}")
        return df

def download_all_data():
    """
    Returns a single DataFrame with columns
    [Date, Ticker, Open, High, Low, Close, Volume]
    """
    return DataCollector().load_or_download()
