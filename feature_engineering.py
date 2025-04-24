# feature_engineering.py
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import config

class FeatureEngineer:
    def __init__(self, raw_data):
        self.raw = raw_data

    def make_features(self):
        features, targets, dates = {}, {}, {}

        for ticker, df in self.raw_data.items():
            # 1) Determine which price column to use
            if 'Close' in df.columns:
                price_col = 'Close'  # standard column from yf.download without adjust
            elif 'Adj Close' in df.columns:
                price_col = 'Adj Close'  # fallback if auto_adjust was False
            else:
                raise KeyError(f"Ticker {ticker}: no 'Close' or 'Adj Close' column found.")  # guard clause

            # 2) Drop rows missing the chosen price
            df_clean = df.copy().dropna(subset=[price_col]) 

            # 3) Calculate returns, moving averages, volatility, RSI, momentum
            df_feat = pd.DataFrame(index=df_clean.index)
            df_feat['returns_1d']    = df_clean[price_col].pct_change(1)
            df_feat['returns_5d']    = df_clean[price_col].pct_change(5)
            df_feat['ma_10d']        = df_clean[price_col].rolling(10, min_periods=1).mean()
            df_feat['volatility_5d'] = df_clean[price_col].pct_change().rolling(5, min_periods=1).std()
            # ... add other indicators as needed ...

            # 4) Generate binary target: next-day up/down
            df_target = (df_clean[price_col].pct_change().shift(-1) > 0).astype(int)

            # 5) Clean infinities and fill missing
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            df_target = df_target.dropna()

            # 6) Align features & targets
            common_idx = df_feat.index.intersection(df_target.index)
            features[ticker] = df_feat.loc[common_idx]
            targets[ticker]  = df_target.loc[common_idx]
            dates[ticker]    = common_idx

        return features, targets, dates

    @staticmethod
    def _rsi(prices, window):
        delta = prices.diff()
        up   = delta.clip(lower=0).rolling(window).mean()
        down = -delta.clip(upper=0).rolling(window).mean()
        rs   = up/down.replace(0, np.nan)
        return 100 - 100/(1+rs)
