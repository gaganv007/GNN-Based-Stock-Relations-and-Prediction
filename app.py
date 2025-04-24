# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from train import train_gnn
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from utils import load_model

st.title("GNN Stock Predictor")
ticker = st.selectbox("Ticker", config.STOCKS)
if st.button("Retrain & Predict"):
    feats,tars = FeatureEngineer().generate_features()
    graphs = construct_graph(feats)
    train_gnn("GCN",feats,tars,graphs)
    model = load_model("GCN")
    df = feats[ticker]
    # dummy predict on last node
    x = torch.tensor(df.values[-1:],dtype=torch.float)
    ei= torch.tensor(list(graphs[0].edges()),dtype=torch.long).t()
    out = model(x, ei)
    pred = out.argmax().item()
    st.write("Prediction:", "Up" if pred else "Down")
