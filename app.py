import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import config
from data_collection import download_all_data
from feature_engineering import prepare_features
from graph_construction import construct_graph
from utils import load_model

st.set_page_config(page_title="GNN Stock Relations and Predictions", layout="wide")
st.title("GNN Stock Relations and Predictions")

@st.cache_data(show_spinner=False)
def init_pipeline():
    download_all_data()
    feats, tgts = prepare_features()
    G = construct_graph(feats)
    return feats, tgts, G

feats, tgts, G = init_pipeline()

common_dates = set.intersection(*(set(df.index) for df in feats.values()))
last_date     = sorted(common_dates)[-1]
st.subheader(f"Last available date: **{last_date.date()}**")

model_name = st.sidebar.selectbox("Model", config.MODELS)
ticker     = st.sidebar.selectbox("Ticker", config.STOCKS)

cum = (feats[ticker]
       .filter(like="returns_1d")
       .cumsum() + 1)
st.line_chart(cum)

feat_len = next(iter(feats.values())).shape[1]
X_rows   = [
    feats[t].loc[last_date].values
    if (t in feats and last_date in feats[t].index)
    else np.zeros(feat_len)
    for t in config.STOCKS
]
X_np     = np.vstack(X_rows)
X_tensor = torch.tensor(X_np, dtype=torch.float)

idx_map = {t: i for i, t in enumerate(config.STOCKS)}
src = []
dst = []
for u, v in G.edges():
    if u in idx_map and v in idx_map:
        src.append(idx_map[u])
        dst.append(idx_map[v])
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
        out = model(X_tensor.to(device), edge_index.to(device))
        if isinstance(out, tuple):
            logits = out[0]
            attn   = out[1] if len(out) > 1 else None
        else:
            logits = out
            attn   = None
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

pred_prob = float(probs[idx_map[ticker]])
direction = "📈 Up" if pred_prob >= 0.5 else "📉 Down"

st.markdown("### Next-Day Prediction")
st.metric(label=ticker, value=direction, delta=f"{pred_prob:.1%}")

df_out = (
    pd.DataFrame({"Ticker": config.STOCKS, "Up Probability": probs})
      .sort_values("Up Probability", ascending=False)
      .reset_index(drop=True)
)
df_out["Up Probability"] = df_out["Up Probability"].map("{:.1%}".format)
st.markdown("### Top 5 Stocks by Predicted Up Probability")
st.table(df_out.head(5))

if attn is not None:
    st.markdown("### Top-Influential Peers (via average attention)")
    w_dict = {}
    weights = attn.mean(dim=1).cpu().numpy()
    for (u, v), w in zip(G.edges(), weights):
        w_dict.setdefault(u, 0.0)
        w_dict.setdefault(v, 0.0)
        w_dict[u] += w
        w_dict[v] += w

    peers = sorted(
        ((peer, weight) for peer, weight in w_dict.items() if peer != ticker),
        key=lambda x: -x[1]
    )[:5]

    for peer, weight in peers:
        st.write(f"- **{peer}**: {weight:.4f}")

    sub_nodes = [ticker] + [p for p, _ in peers]
    subG = G.subgraph(sub_nodes)
    pos  = nx.spring_layout(subG, seed=42)
    fig, ax = plt.subplots()
    nx.draw(subG, pos, with_labels=True, node_size=500, ax=ax)
    widths = [
        (w_dict.get(v, 0.0) + w_dict.get(u, 0.0)) * 2
        for u, v in subG.edges()
    ]
    nx.draw_networkx_edges(subG, pos, width=widths, ax=ax)
    st.pyplot(fig)
