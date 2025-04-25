# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

import config
from data_collection     import download_all_data
from feature_engineering import prepare_features
from graph_construction  import construct_graph
from utils               import load_model
from sklearn.ensemble    import RandomForestClassifier

st.set_page_config(page_title="GNN Stock Predictor", layout="wide")
st.title("📈 GNN Stock Prediction with Explainability")

@st.cache_data
def init_pipeline():
    download_all_data()
    feats, tgts = prepare_features()
    G = construct_graph(feats)
    return feats, tgts, G

feats, tgts, G = init_pipeline()

# Determine last common date
common_dates = set.intersection(*(set(df.index) for df in feats.values()))
last_date = sorted(common_dates)[-1]
st.subheader(f"Last available date: {last_date.date()}")

# Sidebar selectors
model_name = st.sidebar.selectbox("Model", config.MODELS)
ticker     = st.sidebar.selectbox("Ticker", config.STOCKS)

# Plot cumulative returns
cum_returns = feats[ticker]['returns_1d'].loc[:last_date].cumsum() + 1
st.line_chart(cum_returns)

# Build feature tensor / array for last_date
feat_len = feats[ticker].shape[1]
X_rows = [
    feats[t].loc[last_date].values if last_date in feats[t].index 
    else np.zeros(feat_len)
    for t in config.STOCKS
]
X_tensor = torch.tensor(np.vstack(X_rows), dtype=torch.float)
X_np     = np.vstack(X_rows)

# Map tickers to indices
idx_map = {t: i for i, t in enumerate(config.STOCKS)}

# Build edge_index for GNNs
src_idx = [idx_map[u] for u, v in G.edges()]
dst_idx = [idx_map[v] for u, v in G.edges()]
edge_index = torch.tensor([src_idx + dst_idx, dst_idx + src_idx], dtype=torch.long)

# Load model
model = load_model(model_name, in_dim=feat_len, device=None)

# Predict
if isinstance(model, RandomForestClassifier):
    # Sklearn path
    probs     = model.predict_proba(X_np)[:, 1]
    pred_prob = probs[idx_map[ticker]]
    direction = "📈 Up" if pred_prob >= 0.5 else "📉 Down"
    attn      = None
else:
    # PyTorch GNN path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(X_tensor.to(device), edge_index.to(device))
        if isinstance(out, tuple):
            logits, attn = out
        else:
            logits, attn = out, None
        probs     = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred_prob = probs[idx_map[ticker]]
        direction = "📈 Up" if pred_prob >= 0.5 else "📉 Down"

# Display prediction
st.markdown("### Next-Day Prediction")
st.metric(label=ticker, value=direction, delta=f"{pred_prob:.1%}")

# Top-5 by probability
df_out = pd.DataFrame({"Ticker": config.STOCKS, "Up Prob": probs})
df_out = df_out.sort_values("Up Prob", ascending=False).head(5)
st.table(df_out)

# Explainability for GNN
if attn is not None:
    st.markdown("### Top 5 Influential Peers")
    attn_weights = attn.mean(dim=1).cpu().numpy()
    influence = {}
    for (u, v), w in zip(G.edges(), attn_weights):
        if v == ticker:
            influence[u] = influence.get(u, 0.0) + w
        if u == ticker:
            influence[v] = influence.get(v, 0.0) + w
    top_peers = sorted(influence.items(), key=lambda x: -x[1])[:5]
    for peer, w in top_peers:
        st.write(f"- **{peer}**: attention {w:.3f}")

    # Visualize subgraph
    peers = [p for p, _ in top_peers]
    subG = G.subgraph([ticker] + peers)
    pos  = nx.spring_layout(subG)
    fig, ax = plt.subplots()
    nx.draw(subG, pos, with_labels=True, node_size=500, ax=ax)
    edge_ws = []
    for u, v in subG.edges():
        if v == ticker:
            edge_ws.append(influence.get(u, 0.0) * 5)
        elif u == ticker:
            edge_ws.append(influence.get(v, 0.0) * 5)
        else:
            edge_ws.append(1)
    nx.draw_networkx_edges(subG, pos, width=edge_ws, ax=ax)
    st.pyplot(fig)
