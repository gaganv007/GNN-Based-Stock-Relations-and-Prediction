import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def save_model(model, model_name, models_dir="saved_models"):
    """
    Save a PyTorch model (or sklearn) to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), path)

def load_model(model_name, models_dir="saved_models"):
    """
    Load a PyTorch model checkpoint. Assumes you re-create the model with the same architecture.
    """
    from models import get_model
    path = os.path.join(models_dir, f"{model_name}.pt")
    model = get_model(model_name)
    model.load_state_dict(torch.load(path))
    return model

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
    }

def calculate_returns(targets, preds, returns_array):
    """
    Compute trading strategy returns given binary predictions.

    Args:
        targets (array-like): True 0/1 labels (unused here, but for consistency).
        preds (array-like): Model predictions (0=hold, 1=buy).
        returns_array (array-like): Actual percentage returns per sample.

    Returns:
        dict: strategy, benchmark, and excess returns.
    """
    returns = np.asarray(returns_array)
    buys = np.asarray(preds)
    strategy_returns = float((returns * buys).sum())
    benchmark_returns = float(returns.sum())
    excess = strategy_returns - benchmark_returns
    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'excess_returns': excess
    }

def prepare_lstm_data(features_dict, targets_dict, seq_len):
    """
    Flatten dict-of-DataFrames into sequential arrays for LSTM.
    """
    X, y = [], []
    for ticker, feats in features_dict.items():
        tgt = targets_dict[ticker].values
        data = feats.values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(tgt[i+seq_len])
    return np.array(X), np.array(y)

def prepare_random_forest_data(features_dict, targets_dict):
    """
    Flatten dict-of-DataFrames into X, y for RandomForest.
    """
    X = []
    y = []
    for ticker, feats in features_dict.items():
        X.append(feats.values)
        y.append(targets_dict[ticker].values)
    return np.vstack(X), np.hstack(y)

def load_train_test_data(data_dir="data/processed"):
    """
    Load the pickled features/targets and split into train/test.
    """
    feats = pd.read_pickle(os.path.join(data_dir, "features.pkl"))
    tars  = pd.read_pickle(os.path.join(data_dir, "targets.pkl"))
    # Example split: train everything before 2023-01-01
    cutoff = datetime(2023, 1, 1)
    train_feats, train_tars = {}, {}
    test_feats, test_tars   = {}, {}
    for tk in feats:
        df = feats[tk]
        ts = tars[tk]
        train_mask = df.index < cutoff
        train_feats[tk] = df.loc[train_mask]
        train_tars[tk]  = ts[train_mask]
        test_feats[tk]  = df.loc[~train_mask]
        test_tars[tk]   = ts[~train_mask]
    return train_feats, test_feats, train_tars, test_tars
