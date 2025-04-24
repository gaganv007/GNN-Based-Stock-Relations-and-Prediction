import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import config
from data_collection     import download_all_data
from feature_engineering import prepare_features
from graph_construction  import construct_graph
from utils               import load_model

st.set_page_config(page_title="GNN Stock Predictor", layout="wide")
st.title("📈 GNN Stock Prediction")

st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", config.MODELS)
ticker     = st.sidebar.selectbox("Ticker", config.STOCKS)

@st.cache_data(show_spinner=False)
def init_pipeline():
    download_all_data()
    feats, tgts = prepare_features()
    _ = construct_graph(feats)
    return feats, tgts

@st.cache_data(show_spinner=False)
def load_price_history(symbol):
    df = yf.download(
        ticker,
        start=config.START_DATE,
        end=config.END_DATE,
        progress=False,
        auto_adjust=True
    )
    return df["Close"]

feats, tgts = init_pipeline()

common = None
for df in feats.values():
    common = set(df.index) if common is None else common & set(df.index)
if not common:
    st.error("No overlapping dates found in feature data.")
    st.stop()
last_date = sorted(common)[-1]

st.subheader(f"Last available date for prediction: **{last_date.date()}**")

st.markdown(f"### {ticker} Price History")
price_series = load_price_history(ticker)
st.line_chart(price_series)

feat_len = next(iter(feats.values())).shape[1]
X_rows = []
for t in config.STOCKS:
    df = feats.get(t)
    if df is not None and last_date in df.index:
        X_rows.append(df.loc[last_date].values)
    else:
        X_rows.append(np.zeros(feat_len))
X = torch.tensor(np.vstack(X_rows), dtype=torch.float)

edgelist_path = os.path.join(config.GRAPHS_DIR, "correlation_edgelist.csv")
if os.path.exists(edgelist_path):
    edf = pd.read_csv(edgelist_path)
    idx_map = {t: i for i, t in enumerate(config.STOCKS)}
    src, dst = [], []
    for _, row in edf.iterrows():
        s, d = row["source"], row["target"]
        if s in idx_map and d in idx_map:
            src.append(idx_map[s]); dst.append(idx_map[d])
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model_name, in_dim=feat_len, device=device)
model.eval()
with torch.no_grad():
    logits = model(X.to(device), edge_index.to(device))
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

idx_map    = {t: i for i, t in enumerate(config.STOCKS)}
pred_prob  = probs[idx_map[ticker]]
direction  = "📈 Up" if pred_prob >= 0.5 else "📉 Down"
st.markdown("### Next‐Day Prediction")
st.metric(label=f"{ticker}", value=direction, delta=f"{pred_prob:.1%}")

st.markdown("### Top 5 Stocks by Predicted Up Probability")
df_out = (
    pd.DataFrame({"Ticker": config.STOCKS, "Up Probability": probs})
      .sort_values("Up Probability", ascending=False)
      .reset_index(drop=True)
)
df_out["Up Probability"] = df_out["Up Probability"].map("{:.1%}".format)
st.table(df_out.head(5))
