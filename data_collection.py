import os
import time
import random
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import config

def download_all_data():
    # Download stock data
    df = yf.download(
        tickers=config.STOCKS,
        start=config.START_DATE,
        end=config.END_DATE,
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )
    raw_dir = os.path.join(config.DATA_DIR, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    df.to_pickle(os.path.join(raw_dir, 'stock_data.pkl'))

    # Download Google Trends data if enabled
    trends_df = None
    if config.USE_GOOGLE_TRENDS:
        end = datetime.now()
        start = end - timedelta(days=config.GOOGLE_TRENDS_LOOKBACK)
        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        trends = {}
        for stk in config.STOCKS:
            try:
                py = TrendReq(hl='en-US', tz=360)
                py.build_payload([stk], timeframe=timeframe)
                td = py.interest_over_time()
                if 'isPartial' in td.columns:
                    td = td.drop(columns=['isPartial'])
                if not td.empty:
                    trends[stk] = td[stk]
            except Exception:
                pass
            time.sleep(random.uniform(1, 2))

        if trends:
            trends_df = pd.DataFrame(trends)
            trends_df.to_pickle(os.path.join(raw_dir, 'trends_data.pkl'))

    # Merge stock and trends
    raw = pd.read_pickle(os.path.join(raw_dir, 'stock_data.pkl'))
    merged = {}
    for stk in config.STOCKS:
        df_stk = raw[stk].copy()
        if config.USE_GOOGLE_TRENDS and trends_df is not None and stk in trends_df.columns:
            ts = trends_df[stk].resample('D').mean().ffill().reindex(df_stk.index, method='ffill')
            df_stk['GoogleTrend'] = ts
        else:
            df_stk['GoogleTrend'] = 0
        merged[stk] = df_stk

    proc_dir = os.path.join(config.DATA_DIR, 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    pd.to_pickle(merged, os.path.join(proc_dir, 'merged_data.pkl'))
    return merged

if __name__ == '__main__':
    download_all_data()
