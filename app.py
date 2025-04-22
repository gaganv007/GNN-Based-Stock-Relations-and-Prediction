import pandas as pd
import streamlit as st
import torch
import yfinance as yf
import plotly.graph_objects as go

import config
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from utils import load_model
from torch_geometric.data import Data

st.set_page_config(page_title="GNN Stock Prediction", page_icon="📈", layout="wide")

model_name = st.sidebar.selectbox("Select GNN Model", ["GCN","GAT","GraphSAGE","TemporalGNN"])
ticker     = st.sidebar.selectbox("Select Stock",   config.STOCKS)
start_date = st.sidebar.date_input("Start Date",     value=pd.to_datetime(config.START_DATE), max_value=pd.to_datetime(config.END_DATE))
end_date   = st.sidebar.date_input("End Date",       value=pd.to_datetime(config.END_DATE),   min_value=start_date)

# 1) Load OHLC data
@st.cache_data
def load_all(stocks, sd, ed):
    raw = yf.download(tickers=stocks, start=sd, end=ed,
                      group_by="ticker", auto_adjust=True, progress=False)
    out = {}
    for s in stocks:
        out[s] = raw[s].copy() if s in raw.columns.levels[0] else pd.DataFrame()
    return out

price_data = load_all(config.STOCKS, start_date, end_date)

# 2) Show price chart
df = price_data[ticker]
if df.empty:
    st.error(f"No data for {ticker} in that range."); st.stop()

fig = go.Figure([go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"],   close=df["Close"]
)])
fig.update_layout(title=f"{ticker} Price", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# 3) Features & graph
fe = FeatureEngineer(price_data)
feats, _ = fe.generate_features()
gc = construct_graph(features=feats)
G  = gc.build_combined_graph()

# 4) Latest‐date node features
latest = feats[ticker].index[-1]
node2idx = {s:i for i,s in enumerate(config.STOCKS)}

x_list = []
for s in config.STOCKS:
    ser = feats[s].reindex(feats[ticker].index, method="ffill")
    x_list.append(ser.loc[latest].values)
x = torch.tensor(x_list, dtype=torch.float)

edges = []
for u, v in G.edges():
    if u in node2idx and v in node2idx:
        ui, vi = node2idx[u], node2idx[v]
        edges += [(ui, vi), (vi, ui)]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)
data.batch = torch.zeros(x.size(0), dtype=torch.long)

@st.cache_resource
def run_gnn(_data_obj, m):
    model = load_model(m)
    model.eval()
    with torch.no_grad():
        out   = model(_data_obj)             # [1,2]
        prob  = torch.softmax(out, dim=1)[0,1]  # scalar
        pred  = int(prob > 0.5)
    return pred, float(prob)

pred, prob = run_gnn(data, model_name)
dirn = "Up" if pred==1 else "Down"
conf = f"{prob:.1%}"

st.markdown("### Prediction")
st.metric(label=f"{ticker} → {dirn}", value=conf)
