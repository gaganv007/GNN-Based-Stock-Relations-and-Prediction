import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import joblib
import yfinance as yf
from pytrends.request import TrendReq

import config
from models import get_model, LSTMModel
from utils import load_model, prepare_lstm_data, prepare_random_forest_data
from feature_engineering import FeatureEngineer

st.set_page_config(page_title="GNN Stock Prediction", page_icon="📈", layout="wide")

model_name = st.sidebar.selectbox("Select Model", options=config.MODELS)
ticker     = st.sidebar.selectbox("Select Stock", options=config.STOCKS)
start_date = st.sidebar.date_input("Start Date",  value=datetime.now() - timedelta(days=365))
end_date   = st.sidebar.date_input("End Date",    value=datetime.now())

@st.cache_data
def load_stock_data(tkr, sd, ed):
    """Download stock price data."""
    return yf.download(tkr, start=sd, end=ed)

@st.cache_data
def load_trends(tkr, sd, ed):
    """
    Load Google Trends data, handling rate limits gracefully.
    """
    if not config.USE_GOOGLE_TRENDS:
        return None
    try:
        from pytrends.exceptions import TooManyRequestsError
        py = TrendReq(hl='en-US', tz=360)
        timeframe = f"{sd.strftime('%Y-%m-%d')} {ed.strftime('%Y-%m-%d')}"
        py.build_payload([tkr], timeframe=timeframe)
        df = py.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        return df if not df.empty else None
    except TooManyRequestsError:
        st.warning("Google Trends rate limit reached. Skipping trends data.")
        return None
    except Exception as e:
        st.error(f"Error loading Google Trends: {e}")
        return None

@st.cache_resource
def load_model_res(name):
    """
    Load the specified model (RandomForest or LSTM).
    GNN models should be invoked via the CLI pipeline instead.
    """
    if name == 'RandomForest':
        return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))
    if name == 'LSTM':
        m = LSTMModel(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
        m.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, f"{name}.pt")))
        m.eval()
        return m
    st.warning("GNN models must be run via the command-line pipeline (main.py).")
    return None

st.title("📈 GNN Stock Prediction Dashboard")

# Load and display stock price data
df_stock = load_stock_data(ticker, start_date, end_date)
if df_stock.empty:
    st.error("No stock data available for the selected period.")
else:
    fig = go.Figure(data=[go.Candlestick(
        x=df_stock.index,
        open=df_stock['Open'], high=df_stock['High'],
        low=df_stock['Low'], close=df_stock['Close']
    )])
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Load and display Google Trends
    df_trends = load_trends(ticker, start_date, end_date)
    if df_trends is not None:
        st.subheader("Google Trends")
        tfig = go.Figure([go.Scatter(
            x=df_trends.index, y=df_trends[ticker], mode='lines'
        )])
        tfig.update_layout(title=f"Search Interest: {ticker}", xaxis_title="Date", yaxis_title="Interest")
        st.plotly_chart(tfig, use_container_width=True)

    # Feature engineering
    fe = FeatureEngineer({ticker: df_stock})
    feats, targets = fe.generate_features()
    features = feats[ticker]
    if df_trends is not None:
        features['GoogleTrend'] = df_trends.reindex(features.index, method='ffill')[ticker]

    # Model prediction
    model = load_model_res(model_name)
    if model:
        X = features.values
        if model_name == 'RandomForest':
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]
        elif model_name == 'LSTM':
            X_seq, _ = prepare_lstm_data({ticker: features}, {ticker: targets[ticker]}, seq_len=5)
            tensor = torch.tensor(X_seq, dtype=torch.float32)
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[:, 1].detach().numpy()
            preds = (probs > 0.5).astype(int)
        else:
            preds = np.array([])
            probs = np.array([])

        # Display prediction results
        res_df = pd.DataFrame({
            'Date': features.index,
            'Prediction': preds,
            'Probability Up': probs
        })
        st.subheader("Prediction Results")
        st.dataframe(res_df.tail(), use_container_width=True)

        # Next-day metric
        last = res_df.iloc[-1]
        direction = "Up 📈" if last['Prediction'] == 1 else "Down 📉"
        st.metric("Next-Day Prediction", direction, f"{last['Probability Up']*100:.2f}%")
