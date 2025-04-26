# utils.py

import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import config

def save_model(model, name):
    path = os.path.join(config.MODELS_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)

def load_model(name, in_dim=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("TemporalGAT", "GATWithAtt", "GCN", "GraphSAGE"):
        from models import get_model
        net = get_model(name, input_dim=in_dim).to(device)
        net.load_state_dict(torch.load(
            os.path.join(config.MODELS_DIR, f"{name}.pt"),
            map_location=device
        ))
        net.eval()
        return net
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))

def calculate_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob[:,1]) if y_prob is not None else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

def load_train_test_data(feats, tgts, split_date=config.TEST_START_DATE):
    from graph_construction import construct_graph
    G = construct_graph(feats)

    idx_map = {t: i for i, t in enumerate(config.STOCKS + list(config.SECTOR_MAP.keys()))}
    src, dst = [], []
    for u, v in G.edges():
        if u in idx_map and v in idx_map:
            src.append(idx_map[u])
            dst.append(idx_map[v])
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    dates = sorted(next(iter(feats.values())).index)
    train_list, test_list = [], []

    for d in dates:
        X_rows, Y_rows = [], []
        for t in config.STOCKS + list(config.SECTOR_MAP.keys()):
            if t in feats and d in feats[t].index:
                X_rows.append(feats[t].loc[d].values)
                Y_rows.append(int(tgts.get(t, pd.Series()).get(d, 0)))
            else:
                X_rows.append(np.zeros(len(feats[next(iter(feats))].columns)))
                Y_rows.append(0)

        data = Data(
            x = torch.tensor(np.vstack(X_rows), dtype=torch.float),
            edge_index = edge_index,
            y = torch.tensor(Y_rows, dtype=torch.long)
        )

        if pd.to_datetime(d) < pd.to_datetime(split_date):
            train_list.append(data)
        else:
            test_list.append(data)

    return train_list, test_list
