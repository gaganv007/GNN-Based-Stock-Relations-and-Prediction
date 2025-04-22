import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from fredapi import Fred
import requests
from textblob import TextBlob

class FeatureEngineer:
    def __init__(self, price_data):
        """
        price_data: dict ticker -> OHLC DataFrame (from Yahoo Finance)
        """
        self.price_data = price_data
        self.features   = {}
        self.targets    = {}
        self.fred       = Fred(api_key=config.FRED_API_KEY)
        self.news_key   = config.NEWS_API_KEY

    def _fetch_macro(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        for series_id, name in config.MACRO_SERIES.items():
            s = self.fred.get_series(
                series_id,
                observation_start=dates.min(),
                observation_end=dates.max()
            )
            df[name] = s.reindex(dates).ffill()
        return df

    def _fetch_etf_flows(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        for etf in config.ETF_TICKERS:
            pdf = self.price_data.get(etf)
            if pdf is not None and "Volume" in pdf:
                df[f"{etf}_Vol"] = pdf["Volume"].reindex(dates).ffill()
        return df

    def _fetch_commodities(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame(index=dates)
        for cmd in config.COMMODITY_TICKERS:
            pdf = self.price_data.get(cmd)
            if pdf is not None and "Close" in pdf:
                df[f"{cmd}_Close"] = pdf["Close"].reindex(dates).ffill()
        return df

    def _fetch_news_sentiment(self, ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
        sentiments = []
        for date in dates:
            frm = date.strftime("%Y-%m-%d")
            to  = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            url = (
                "https://newsapi.org/v2/everything?"
                f"q={ticker}&from={frm}&to={to}"
                f"&sortBy=popularity&apiKey={self.news_key}"
            )
            try:
                res = requests.get(url, timeout=5).json()
                arts = res.get("articles", [])
            except Exception:
                arts = []
            scores = []
            for art in arts:
                txt = (art.get("title","") + " " + art.get("description","")).strip()
                if txt:
                    scores.append(TextBlob(txt).sentiment.polarity)
            sentiments.append(np.nanmean(scores) if scores else 0.0)
        return pd.Series(sentiments, index=dates, name="NewsSentiment")

    def generate_features(self):
        # Master dates from any non-empty ticker
        sample = next((df for df in self.price_data.values() if not df.empty), None)
        if sample is None:
            raise ValueError("No price data available.")
        dates = sample.index

        # Pre‑fetch shared series
        macro_df = self._fetch_macro(dates)
        etf_df   = self._fetch_etf_flows(dates)
        cmd_df   = self._fetch_commodities(dates)

        for tk, pdf in tqdm(self.price_data.items(), desc="Feat Eng"):
            if pdf.empty:
                continue
            d = pdf.copy()

            # --- TECHNICAL INDICATORS ---
            d["returns_1d"]    = d["Close"].pct_change(1)
            d["returns_5d"]    = d["Close"].pct_change(5)
            d["ma_10d"]        = d["Close"].rolling(10, min_periods=1).mean()
            d["volatility_5d"] = d["Close"].pct_change().rolling(5, min_periods=1).std()
            # (add RSI, momentum, etc.)

            # --- MERGE MACRO / ETF / COMMODITY ---
            d = d.merge(macro_df,   left_index=True, right_index=True, how="left")
            d = d.merge(etf_df,     left_index=True, right_index=True, how="left")
            d = d.merge(cmd_df,     left_index=True, right_index=True, how="left")

            # --- NEWS SENTIMENT ---
            news_ser = self._fetch_news_sentiment(tk, dates)
            d = d.merge(news_ser, left_index=True, right_index=True, how="left")

            # --- NEXT‑DAY UP/DOWN TARGET ---
            d["target"] = (d["Close"].pct_change().shift(-1) > 0).astype(int)

            # --- CLEAN & STORE ---
            d = d.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
            feat_cols = [c for c in d.columns if c not in ["Open","High","Low","Close","Volume","target"]]
            self.features[tk] = d[feat_cols]
            self.targets[tk]  = d["target"]

        return self.features, self.targets
