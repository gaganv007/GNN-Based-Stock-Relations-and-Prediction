import os, pickle, pandas as pd, numpy as np, config

FEAT_PKL = os.path.join(config.PROCESSED_DIR, "features.pkl")
TGT_PKL  = os.path.join(config.PROCESSED_DIR, "targets.pkl")

def add_bollinger_bands(close, window=20, k=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma + k*std, ma - k*std

def add_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def prepare_features():
    with open(os.path.join(config.RAW_DIR, "stock_data.pkl"), "rb") as f:
        raw = pickle.load(f)

    feats, tgts = {}, {}
    for t, df in raw.items():
        if t not in config.STOCKS or "Close" not in df.columns:
            continue
        df = df.sort_index()
        close = df["Close"]
        volume = df.get("Volume", pd.Series(index=close.index, data=0))
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

        for w in config.SMA_WINDOWS:
            f[f"sma_{w}"] = close.rolling(w, min_periods=1).mean()

        tp = (df["High"] + df["Low"] + close)/3
        mf = tp * volume
        pos = mf.where(tp.diff()>0, 0).rolling(config.MFI_WINDOW, min_periods=1).sum()
        neg = mf.where(tp.diff()<0, 0).rolling(config.MFI_WINDOW, min_periods=1).sum()
        f["mfi"] = 100 - 100/(1 + pos/(neg.replace(0, np.nan))).fillna(50)

        bb_u, bb_l = add_bollinger_bands(close)
        macd, sig = add_macd(close)
        f["bb_upper"], f["bb_lower"] = bb_u, bb_l
        f["macd"], f["macd_signal"] = macd, sig

        f["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        f["vroc_5d"] = volume.pct_change(5)

        o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
        f["is_doji"] = ((c-o).abs() < 0.001*(h-l)).astype(int)
        f["is_bull_engulf"] = ((c>o.shift())&(o<c.shift())).astype(int)

        f = f.replace([np.inf,-np.inf], np.nan).dropna(how="any")
        tgt = (close.pct_change(1).shift(-1)>0).astype(int)
        tgt = tgt.reindex(f.index).fillna(0).astype(int)

        if not f.empty:
            feats[t], tgts[t] = f, tgt

    feats_w, tgts_w = {}, {}
    W = config.FEATURE_WINDOW
    for t, df in feats.items():
        if len(df) <= W: continue
        arr = np.hstack([df.shift(i).iloc[W:].values for i in range(W)])
        cols = [f"{col}_t-{i}" for i in range(W) for col in df.columns]
        feats_w[t] = pd.DataFrame(arr, index=df.index[W:], columns=cols)
        tgts_w[t]  = tgts[t].iloc[W:]

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(FEAT_PKL, "wb") as f: pickle.dump(feats_w, f)
    with open(TGT_PKL, "wb")  as f: pickle.dump(tgts_w, f)
    return feats_w, tgts_w