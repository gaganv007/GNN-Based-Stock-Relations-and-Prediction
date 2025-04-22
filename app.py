import os
import streamlit as st
import pandas as pd
import torch
from torch_geometric.data import Data
import plotly.graph_objects as go
import yfinance as yf

import config
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from utils import load_model

st.set_page_config(page_title="GNN Stock Prediction", page_icon="📈", layout="wide")

# Sidebar
model_name = st.sidebar.selectbox("Select GNN Model", ["GCN","GAT","GraphSAGE","TemporalGNN"])
ticker     = st.sidebar.selectbox("Select Stock", config.STOCKS)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(config.START_DATE))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime(config.END_DATE))

@st.cache_data
def load_all_stock_data(stocks, sd, ed):
    raw = yf.download(tickers=stocks, start=sd, end=ed,
                      group_by="ticker", auto_adjust=True, progress=False)
    out = {}
    for stk in stocks:
        out[stk] = raw[stk].copy() if stk in raw.columns.levels[0] else pd.DataFrame()
    return out

# 1️⃣ Load and show price chart for selected ticker
data_dict = load_all_stock_data(config.STOCKS, start_date, end_date)
df = data_dict[ticker]
if df.empty:
    st.error(f"No data for {ticker} in the chosen range."); st.stop()

fig = go.Figure([go.Candlestick(x=df.index,
                                open=df["Open"], high=df["High"],
                                low=df["Low"],   close=df["Close"])])
fig.update_layout(title=f"{ticker} Price History", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# 2️⃣ Feature engineering & graph construction
fe = FeatureEngineer(data_dict)
feats, targets = fe.generate_features()
gc = construct_graph(features=feats)
G = gc.build_combined_graph()

# 3️⃣ Build node features tensor for latest date
latest = feats[ticker].index[-1]
node2idx = {stk: i for i, stk in enumerate(config.STOCKS)}

x_list = []
for stk in config.STOCKS:
    ser = feats[stk].reindex(feats[ticker].index, method="ffill")
    x_list.append(ser.loc[latest].values)
x = torch.tensor(x_list, dtype=torch.float)

# 4️⃣ Build edge_index
edge_list = []
for u, v in G.edges():
    if u in node2idx and v in node2idx:
        ui, vi = node2idx[u], node2idx[v]
        edge_list += [(ui, vi), (vi, ui)]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)
data.batch = torch.zeros(x.size(0), dtype=torch.long)

# 5️⃣ Run GNN inference (single-graph)
@st.cache_resource
def run_gnn(_data_obj, model_name):
    model = load_model(model_name)
    model.eval()
    with torch.no_grad():
        out   = model(_data_obj)               # [1,2]
        probs = torch.softmax(out, dim=1)[0,1]  # scalar
        pred  = int(probs > 0.5)
    return pred, float(probs)

pred, prob = run_gnn(data, model_name)
direction = "Up" if pred == 1 else "Down"
confidence = f"{prob:.1%}"

st.markdown("### Prediction")
st.metric(label=f"{ticker} → {direction}", value=confidence)
