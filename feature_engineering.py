# feature_engineering.py

import os
import pandas as pd
import numpy as np
import config

TECHNICAL_INDICATORS = {
    "returns_1d": 1,
    "returns_5d": 5,
    "ma_5d": 5,
    "ma_10d": 10,
    "ma_20d": 20,
    "volatility_5d": 5,
    "volatility_10d": 10,
    "rsi_14d": 14,
    "momentum_5d": 5,
    "momentum_10d": 10,
}

def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def prepare_features(raw_csv: str, start_date: str, end_date: str):
    """
    Reads raw_csv, filters to date range, pivots per-ticker,
    computes all technical indicators + target, returns:
      feats: dict[ticker -> DataFrame of features indexed by Date]
      targets: dict[ticker -> Series of 0/1 next-day up/down]
    Also caches to processed/*.csv
    """
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    # load & filter
    df = pd.read_csv(raw_csv, parse_dates=["Date"])
    df = df[df["Date"].between(start_date, end_date)]
    # pivot so we can compute on each ticker
    feats = {}
    targets = {}

    for t in config.STOCKS:
        sub = df[df["Ticker"] == t].set_index("Date").sort_index()
        if sub.empty:
            continue
        f = pd.DataFrame(index=sub.index)
        close = sub["Close"]
        # technicals
        for name, w in TECHNICAL_INDICATORS.items():
            if name.startswith("returns"):
                f[name] = close.pct_change(w)
            elif name.startswith("ma"):
                f[name] = close.rolling(w).mean()
            elif name.startswith("volatility"):
                f[name] = close.pct_change().rolling(w).std()
            elif name.startswith("rsi"):
                f[name] = calculate_rsi(close, w)
            elif name.startswith("momentum"):
                f[name] = close.diff(w)
        # drop initial NaNs
        f = f.dropna(how="any")
        # target: next-day up?
        tgt = (close.pct_change().shift(-1) > 0).astype(int).reindex(f.index)
        feats[t] = f
        targets[t] = tgt

        # cache per‐ticker
        f.to_csv(os.path.join(config.PROCESSED_DIR, f"feat_{t}.csv"))
        tgt.to_csv(os.path.join(config.PROCESSED_DIR, f"tgt_{t}.csv"), header=["target"])

    print(f"💾 Features  → {config.PROCESSED_DIR}/feat_*.csv")
    print(f"💾 Targets   → {config.PROCESSED_DIR}/tgt_*.csv")
    return feats, targets
