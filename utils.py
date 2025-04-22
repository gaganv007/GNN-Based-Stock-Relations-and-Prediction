import os
import joblib
import torch
import numpy as np
import config
from models import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def save_model(model, name):
    """
    Save a PyTorch model's state dictionary.
    """
    path = os.path.join(config.MODELS_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)

def load_model(name):
    """
    Load a PyTorch model by name.
    """
    model = get_model(name, config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    path = os.path.join(config.MODELS_DIR, f"{name}.pt")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics.
    Returns a dict with accuracy, precision, recall, and F1 score.
    """
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0)
    }

def prepare_random_forest_data(features, targets):
    """
    Prepare feature matrix X and target vector y for RandomForest.
    Splits into train/test and returns X_train, X_test, y_train, y_test.
    """
    X_list, y_list = [], []
    for ticker, df in features.items():
        X_list.append(df.values)
        y_list.append(targets[ticker].values)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def prepare_lstm_data(features, targets, seq_len=5):
    """
    Prepare sequences of length `seq_len` for LSTM.
    Returns X (samples, seq_len, features) and y (samples, ).
    """
    X, y = [], []
    for ticker, df in features.items():
        arr = df.values
        for i in range(len(arr) - seq_len):
            X.append(arr[i:i+seq_len])
            y.append(targets[ticker].iloc[i + seq_len])
    return np.array(X), np.array(y)
