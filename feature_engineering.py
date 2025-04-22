import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config

class FeatureEngineer:
    def __init__(self, data=None):
        """
        data: dict of {ticker: DataFrame}, where each DataFrame has at least 'Close'.
              If None, load from processed/merged_data.pkl.
        """
        if data is None:
            path = os.path.join(config.DATA_DIR, 'processed', 'merged_data.pkl')
            data = pd.read_pickle(path)
        self.data = data
        self.features = {}
        self.targets  = {}

    def calculate_returns(self, df, w):
        return df['Close'].pct_change(w)

    def calculate_moving_average(self, df, w):
        return df['Close'].rolling(w, min_periods=1).mean()

    def calculate_volatility(self, df, w):
        return df['Close'].pct_change().rolling(w, min_periods=1).std()

    def calculate_rsi(self, df, w):
        delta = df['Close'].diff()
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        ma_up   = up.rolling(w, min_periods=1).mean()
        ma_down = down.rolling(w, min_periods=1).mean()
        rs      = ma_up / ma_down.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def calculate_momentum(self, df, w):
        return df['Close'].diff(w)

    def generate_features(self):
        """
        For each ticker:
        - Compute all configured technical indicators
        - Generate a binary 'target' column for next-day up/down
        - Select only the indicator columns (and GoogleTrend if present)
        """
        for stk, df in tqdm(self.data.items(), desc="Feat Eng"):
            d = df.copy()

            # Compute indicators
            for ind, w in config.TECHNICAL_INDICATORS.items():
                if 'returns' in ind:
                    d[ind] = self.calculate_returns(df, w)
                elif 'ma' in ind:
                    d[ind] = self.calculate_moving_average(df, w)
                elif 'volatility' in ind:
                    d[ind] = self.calculate_volatility(df, w)
                elif 'rsi' in ind:
                    d[ind] = self.calculate_rsi(df, w)
                elif 'momentum' in ind:
                    d[ind] = self.calculate_momentum(df, w)

            # Next-day up/down target
            d['target'] = (d['Close'].pct_change().shift(-1) > 0).astype(int)

            # Drop any rows with NaNs
            d = d.dropna()

            # Build feature column list
            cols = list(config.TECHNICAL_INDICATORS.keys())
            if 'GoogleTrend' in d.columns:
                cols.append('GoogleTrend')

            # Subset and store
            self.features[stk] = d[cols]
            self.targets[stk]  = d['target']

        return self.features, self.targets
