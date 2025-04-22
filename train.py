import os
import torch
import joblib
import config
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMModel
from utils import save_model, calculate_metrics, prepare_random_forest_data, prepare_lstm_data

def train_rf(features, targets):
    X_train, X_test, y_train, y_test = prepare_random_forest_data(features, targets)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(config.MODELS_DIR, 'RandomForest.joblib'))
    y_pred = clf.predict(X_test)
    return calculate_metrics(y_test, y_pred)

def train_lstm(features, targets):
    X, y = prepare_lstm_data(features, targets)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LSTMModel(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    opt   = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        # batch training
        for i in range(0, len(X_train), config.BATCH_SIZE):
            xb = torch.tensor(X_train[i:i+config.BATCH_SIZE], dtype=torch.float32)
            yb = torch.tensor(y_train[i:i+config.BATCH_SIZE], dtype=torch.long)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    save_model(model, 'LSTM')
    model.eval()
    xb = torch.tensor(X_test, dtype=torch.float32)
    preds = model(xb).argmax(dim=1).detach().numpy()
    return calculate_metrics(y_test, preds)
