import pandas as pd
import numpy as np
from config import TECHNICAL_INDICATORS

def compute_technical_features(price_df: pd.DataFrame):
    """
    Given price_df (dates × tickers), return a dict[ticker] → feature‐DataFrame.
    """
    feat_dict = {}
    for ticker in price_df.columns:
        d = pd.DataFrame(index=price_df.index)
        d["Price"] = price_df[ticker]
        # returns
        d["returns_1d"] = d["Price"].pct_change(TECHNICAL_INDICATORS["returns_1d"])
        d["returns_5d"] = d["Price"].pct_change(TECHNICAL_INDICATORS["returns_5d"])
        # moving averages
        d["ma_10d"]     = d["Price"].rolling(TECHNICAL_INDICATORS["ma_10d"]).mean()
        d["ma_20d"]     = d["Price"].rolling(TECHNICAL_INDICATORS["ma_20d"]).mean()
        # volatility (std of returns)
        d["volatility_10d"] = d["Price"].pct_change().rolling(TECHNICAL_INDICATORS["volatility_10d"]).std()
        # RSI
        delta = d["Price"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up   = up.ewm(span=TECHNICAL_INDICATORS["rsi_14d"], adjust=False).mean()
        roll_down = down.ewm(span=TECHNICAL_INDICATORS["rsi_14d"], adjust=False).mean()
        RS = roll_up / roll_down
        d["rsi_14d"] = 100 - (100 / (1 + RS))
        # target: next‐day up/down
        d["target"] = (d["Price"].pct_change().shift(-1) > 0).astype(int)
        # clean missing
        d = d.dropna()
        feat = d.drop(columns=["Price"])
        feat_dict[ticker] = feat
        print(f"  • Features for {ticker}: {feat.shape[0]} rows, {feat.shape[1]} cols")
    return feat_dict
