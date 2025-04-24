import os
import pickle
import pandas as pd
import numpy as np
import config

FEAT_PKL = os.path.join(config.PROCESSED_DIR, "features.pkl")
TGT_PKL  = os.path.join(config.PROCESSED_DIR, "targets.pkl")

def prepare_features():
    raw_path = os.path.join(config.RAW_DIR, "stock_data.pkl")
    with open(raw_path, "rb") as f:
        raw_dict = pickle.load(f)

    feats, tgts = {}, {}
    for ticker, df in raw_dict.items():
        if ticker not in config.STOCKS:
            continue

        df = df.sort_index()
        if "Close" not in df.columns:
            continue

        close = df["Close"]
        f = pd.DataFrame(index=close.index)

        for name, w in config.TECHNICAL_INDICATORS.items():
            if name.startswith("returns"):
                f[name] = close.pct_change(w)
            elif name.startswith("ma"):
                f[name] = close.rolling(w, min_periods=1).mean()
            elif name.startswith("volatility"):
                f[name] = close.pct_change().rolling(w, min_periods=1).std()
            elif name.startswith("rsi"):
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(w, min_periods=1).mean()
                loss  = (-delta).clip(lower=0).rolling(w, min_periods=1).mean()
                rs    = gain / loss.replace(0, np.nan)
                f[name] = (100 - (100 / (1 + rs))).fillna(50)
            elif name.startswith("momentum"):
                f[name] = close.diff(w)

        # drop infinite / na
        f = f.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        # target: next-day up/down
        tgt = (close.pct_change(1).shift(-1) > 0).astype(int)
        tgt = tgt.reindex(f.index).fillna(0).astype(int)

        if not f.empty:
            feats[ticker] = f
            tgts[ticker]  = tgt

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(FEAT_PKL, "wb") as f:
        pickle.dump(feats, f)
    with open(TGT_PKL, "wb") as f:
        pickle.dump(tgts, f)
    print(f"💾 Features  → {FEAT_PKL}")
    print(f"💾 Targets   → {TGT_PKL}")
    return feats, tgts
