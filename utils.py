# utils.py

import torch
import joblib
import pandas as pd
import numpy as np
from torch_geometric.data import Data

import config

def save_model(model, name):
    path = f"{config.MODELS_DIR}/{name}.pt"
    torch.save(model.state_dict(), path)

def load_model(name, in_dim, device=None):
    device = device or config.DEVICE
    net = {
        "GCN":    lambda: __import__("models").models.GCN(in_dim),
        "GAT":    lambda: __import__("models").models.GAT(in_dim),
        "GraphSAGE": lambda: __import__("models").models.GraphSAGE(in_dim),
        "TemporalGNN": lambda: __import__("models").models.TemporalGNN(in_dim)
    }[name]()
    net.load_state_dict(torch.load(f"{config.MODELS_DIR}/{name}.pt", map_location=device))
    return net.to(device)

def load_train_test_data(feats: dict, targets: dict, split_date="2018-01-01"):
    """
    Build PyG datasets: train on dates < split_date, test on >=.
    Returns (train_list, test_list) of Data objects.
    """
    # build edgelist
    edgelist = pd.read_csv(f"{config.GRAPHS_DIR}/correlation_edgelist.csv")
    src = torch.tensor(edgelist.source.values, dtype=torch.long)
    dst = torch.tensor(edgelist.target.values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # node ordering = config.STOCKS
    idx_map = {t:i for i,t in enumerate(config.STOCKS)}
    # per-day graph snapshot
    dates = sorted(list(next(iter(feats.values())).index))
    train_list, test_list = [], []
    for d in dates:
        x = []
        y = []
        for t in config.STOCKS:
            if d in feats[t].index:
                x.append(feats[t].loc[d].values)
                y.append(targets[t].loc[d])
            else:
                x.append(np.zeros(len(feats[t].columns)))
                y.append(0)
        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long)
        )
        if d < pd.to_datetime(split_date):
            train_list.append(data)
        else:
            test_list.append(data)
    return train_list, test_list

def calculate_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob[:,1]) if y_prob is not None else 0.0
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":roc}
