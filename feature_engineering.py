import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config

class FeatureEngineer:
    def __init__(self, data=None):
        """
        data: dict(ticker -> DataFrame). If None, loads from data/processed/merged_data.pkl
        """
        if data is None:
            path = os.path.join(config.DATA_DIR, "processed", "merged_data.pkl")
            data = pd.read_pickle(path)
        self.data     = data
        self.features = {}
        self.targets  = {}

    def calculate_returns(self, df, w):
        return df["Close"].pct_change(w)

    def calculate_moving_average(self, df, w):
        return df["Close"].rolling(w, min_periods=1).mean()

    def calculate_volatility(self, df, w):
        return df["Close"].pct_change().rolling(w, min_periods=1).std()

    def calculate_rsi(self, df, w):
        delta = df["Close"].diff()
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        ma_up   = up.rolling(w, min_periods=1).mean()
        ma_down = down.rolling(w, min_periods=1).mean()
        rs      = ma_up / ma_down.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def generate_features(self):
        """
        For each ticker, computes all technical indicators and
        a next‐day up/down target, then drops NaNs.
        """
        for stk, df in tqdm(self.data.items(), desc="Feat Eng"):
            d = df.copy()

            # Indicators
            for ind, w in config.TECHNICAL_INDICATORS.items():
                if "returns" in ind:
                    d[ind] = self.calculate_returns(df, w)
                elif "ma" in ind:
                    d[ind] = self.calculate_moving_average(df, w)
                elif "volatility" in ind:
                    d[ind] = self.calculate_volatility(df, w)
                elif "rsi" in ind:
                    d[ind] = self.calculate_rsi(df, w)

            # Target: next‐day up/down
            d["target"] = (d["Close"].pct_change().shift(-1) > 0).astype(int)

            # Drop NaNs
            d = d.dropna()

            cols = list(config.TECHNICAL_INDICATORS.keys())
            self.features[stk] = d[cols]
            self.targets[stk]  = d["target"]

        return self.features, self.targets
