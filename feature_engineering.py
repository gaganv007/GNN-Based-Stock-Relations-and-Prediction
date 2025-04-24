import os
import pandas as pd
import numpy as np
import config

FEAT_PKL = os.path.join(config.PROCESSED_DIR, "features.pkl")
TGT_PKL  = os.path.join(config.PROCESSED_DIR, "targets.pkl")

def prepare_features():
    """
    Read raw stock_data.pkl (with MultiIndex columns),
    compute technical indicators & binary targets,
    cache to PROCESSED_DIR, and return (feats_dict, tgts_dict).
    """
    raw_path = os.path.join(config.RAW_DIR, "stock_data.pkl")
    raw = pd.read_pickle(raw_path)

    # Ensure we have a MultiIndex (ticker, field)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex columns in {raw_path}, got {raw.columns}")

    feats, tgts = {}, {}
    for t in config.STOCKS:
        if t not in raw.columns.levels[0]:
            continue

        df = raw[t].copy().sort_index()
        f  = pd.DataFrame(index=df.index)
        close = df["Close"]

        # technical indicators
        for name, w in config.TECHNICAL_INDICATORS.items():
            if name.startswith("returns"):
                f[name] = close.pct_change(w)
            elif name.startswith("ma"):
                f[name] = close.rolling(w, min_periods=w).mean()
            elif name.startswith("volatility"):
                f[name] = close.pct_change().rolling(w, min_periods=w).std()
            elif name.startswith("rsi"):
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(w, min_periods=1).mean()
                loss  = (-delta).clip(lower=0).rolling(w, min_periods=1).mean()
                rs    = gain / loss.replace(0, np.nan)
                f[name] = (100 - (100 / (1 + rs))).fillna(50)
            elif name.startswith("momentum"):
                f[name] = close.diff(w)

        # drop any NaNs/inf after computing
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        # target: next‐day up/down
        tgt = (
            (close.pct_change(1).shift(-1) > 0)
            .astype(int)
            .reindex(f.index)
            .fillna(0)
            .astype(int)
        )

        feats[t] = f
        tgts[t] = tgt

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    pd.to_pickle(feats, FEAT_PKL)
    pd.to_pickle(tgts,  TGT_PKL)
    print(f"💾 Features  → {FEAT_PKL}")
    print(f"💾 Targets   → {TGT_PKL}")
    return feats, tgts
