# feature_engineering.py
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import config

class FeatureEngineer:
    def __init__(self, raw_data):
        """
        raw_data: dict ticker -> DataFrame with Open/High/Low/Close/Volume/return_1d
        """
        self.raw = raw_data

    def make_features(self):
        feats, targets, dates = {}, {}, {}
        for tkr, df in self.raw.items():
            df = df.copy().dropna(subset=["Close"])
            df["ma_5"]   = df["Close"].rolling(5).mean()
            df["ma_20"]  = df["Close"].rolling(20).mean()
            df["vol_5"]  = df["Close"].pct_change().rolling(5).std()
            df["rsi_14"] = self._rsi(df["Close"], 14)
            df = df.dropna()
            X = df[["ma_5","ma_20","vol_5","rsi_14"]].values
            X = StandardScaler().fit_transform(X)
            feats[tkr] = X
            # binary target: up/down
            y = (df["return_1d"]>0).astype(int).values
            targets[tkr] = y
            dates[tkr] = df.index
        return feats, targets, dates

    @staticmethod
    def _rsi(prices, window):
        delta = prices.diff()
        up   = delta.clip(lower=0).rolling(window).mean()
        down = -delta.clip(upper=0).rolling(window).mean()
        rs   = up/down.replace(0, np.nan)
        return 100 - 100/(1+rs)
