import os
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from fredapi import Fred
import aiohttp
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

DATA_DIR = config.DATA_DIR
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

class DataCollector:
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        fred_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_key) if fred_key else None

    def download_price_data(self, tickers=None, start_date=None, end_date=None):
        """Download or load cached stock price data."""
        path = os.path.join(RAW_DIR, "stock_data.pkl")
        if os.path.exists(path):
            print("  • Loading cached stock data")
            return pd.read_pickle(path)

        print("  • Downloading stock data")
        tickers    = tickers    or config.STOCKS
        start_date = start_date or config.START_DATE
        end_date   = end_date   or config.END_DATE

        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        data.to_pickle(path)
        print(f"  • Saved stock data → {path}")
        return data

    async def _fetch_news(self, session, query, frm, to):
        url    = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "from":     frm,
            "to":       to,
            "apiKey":   self.news_api_key,
            "pageSize": 100
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            j = await resp.json()
            return j.get("articles", [])

    def download_news_data(self, queries=None, days_lookback=7):
        """Fetch or load cached NewsAPI articles in parallel."""
        path = os.path.join(RAW_DIR, "news_data.pkl")
        if os.path.exists(path):
            print("  • Loading cached news data")
            return pd.read_pickle(path)

        print("  • Downloading news data (async)…")
        queries = queries or config.STOCKS
        to_date   = datetime.utcnow().strftime("%Y-%m-%d")
        from_date = (datetime.utcnow() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")

        articles = []

        async def _gather():
            async with aiohttp.ClientSession() as sess:
                tasks = [self._fetch_news(sess, q, from_date, to_date) for q in queries]
                for result in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(result, Exception):
                        print("    ! NewsAPI error:", result)
                    else:
                        articles.extend(result)

        asyncio.run(_gather())
        df = pd.DataFrame(articles)
        df.to_pickle(path)
        print(f"  • Saved news data → {path}")
        return df

    def download_fred_data(self, series_ids=None):
        """Fetch or load cached FRED series as DataFrame."""
        path = os.path.join(RAW_DIR, "fred_data.pkl")
        if os.path.exists(path):
            print("  • Loading cached FRED data")
            return pd.read_pickle(path)

        if not self.fred or not series_ids:
            print("  • No FRED key or no series IDs; skipping FRED data")
            return None

        print("  • Downloading FRED data")
        df = pd.DataFrame({sid: self.fred.get_series(sid) for sid in series_ids})
        df.to_pickle(path)
        print(f"  • Saved FRED data → {path}")
        return df

    def download_all_data(self):
        """Helper to run all three downloads in sequence."""
        return {
            "prices": self.download_price_data(),
            "news":   self.download_news_data(),
            "fred":   self.download_fred_data(getattr(config, "FRED_SERIES_IDS", []))
        }