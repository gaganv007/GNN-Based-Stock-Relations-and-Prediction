import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import yfinance as yf

import config
from data_collection     import download_all_data
from feature_engineering import prepare_features
from graph_construction  import construct_graph
from utils               import load_model
from sklearn.ensemble    import RandomForestClassifier

st.set_page_config(page_title="GNN Stock Predictor", layout="wide")
st.title("📈 GNN Stock Prediction with Explainability")

@st.cache_data(show_spinner=False)
def init_pipeline():
    download_all_data()
    feats, tgts = prepare_features()
    G = construct_graph(feats)
    return feats, tgts, G

@st.cache_data(show_spinner=False)
def load_price_history(symbol):
    df = yf.download(
        symbol,
        start=config.START_DATE,
        end=config.END_DATE,
        progress=False,
        auto_adjust=True
    )
    return df["Close"]

feats, tgts, G = init_pipeline()

common = set.intersection(*(set(df.index) for df in feats.values()))
last_date = sorted(common)[-1]
st.subheader(f"Last available date: {last_date.date()}")

model_name = st.sidebar.selectbox("Model", config.MODELS)
ticker     = st.sidebar.selectbox("Ticker", config.STOCKS)

st.markdown(f"### {ticker} Price History")
price_series = load_price_history(ticker)
st.line_chart(price_series)

feat_len = feats[ticker].shape[1]
X_rows = [
    feats[t].loc[last_date].values if last_date in feats[t].index
    else np.zeros(feat_len)
    for t in config.STOCKS
]
X_np     = np.vstack(X_rows)
X_tensor = torch.tensor(X_np, dtype=torch.float)

idx = {t: i for i, t in enumerate(config.STOCKS)}
src = [idx[u] for u, v in G.edges() if u in idx and v in idx]
dst = [idx[v] for u, v in G.edges() if u in idx and v in idx]
edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

model = load_model(model_name, in_dim=feat_len, device=None)
if isinstance(model, RandomForestClassifier):
    probs = model.predict_proba(X_np)[:, 1]
    attn  = None
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()
    with torch.no_grad():
        out    = model(X_tensor.to(device), edge_index.to(device))
        logits, attn = out if isinstance(out, tuple) else (out, None)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

pred_prob = probs[idx[ticker]]
direction = "📈 Up" if pred_prob >= 0.5 else "📉 Down"

st.markdown("### Next-Day Prediction")
st.metric(label=ticker, value=direction, delta=f"{pred_prob:.1%}")

df_out = pd.DataFrame({"Ticker": config.STOCKS, "Up Prob": probs})
st.table(df_out.sort_values("Up Prob", ascending=False).head(5))

if attn is not None:
    st.markdown("### Top-Influential Peers")
    w    = attn.mean(dim=1).cpu().numpy()
    infl = {}
    for (u, v), weight in zip(G.edges(), w):
        if v == ticker: infl[u] = infl.get(u, 0) + weight
        if u == ticker: infl[v] = infl.get(v, 0) + weight

    for peer, weight in sorted(infl.items(), key=lambda x: -x[1])[:5]:
        st.write(f"- **{peer}**: {weight:.3f}")

    subG = G.subgraph([ticker] + [p for p in infl][:5])
    pos  = nx.spring_layout(subG)
    fig, ax = plt.subplots()
    nx.draw(subG, pos, with_labels=True, node_size=500, ax=ax)
    widths = [infl.get(v, 1) * 5 for u, v in subG.edges() if v != ticker]
    nx.draw_networkx_edges(subG, pos, width=widths, ax=ax)
    st.pyplot(fig)
