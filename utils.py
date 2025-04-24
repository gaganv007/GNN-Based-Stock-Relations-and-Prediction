import os
import torch
import joblib
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import config
import torch.nn.functional as F

def save_model(model, name):
    path = os.path.join(config.MODELS_DIR, f"{name}.pt")
    # RandomForest saved via joblib if needed
    if hasattr(model, "predict"):
        joblib.dump(model, os.path.join(config.MODELS_DIR, f"{name}.joblib"))
    else:
        torch.save(model.state_dict(), path)

def load_model(name, in_dim=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("GCN", "GAT", "GraphSAGE", "TemporalGNN"):
        from models import get_model
        net = get_model(name, input_dim=in_dim).to(device)
        net.load_state_dict(torch.load(
            os.path.join(config.MODELS_DIR, f"{name}.pt"),
            map_location=device
        ))
        return net
    # RandomForest
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))

def calculate_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_prob[:,1]) if y_prob is not None else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

def load_train_test_data(feats: dict, tgts: dict, split_date: str = config.TEST_START_DATE):
    """
    Build train/test PyG Data lists.
    Edges come from data/graphs/correlation_edgelist.csv, mapped via config.STOCKS.
    """
    # --- load edgelist ---
    edgelist_path = os.path.join(config.GRAPHS_DIR, "correlation_edgelist.csv")
    if os.path.exists(edgelist_path):
        edf    = pd.read_csv(edgelist_path)
        idx_map= {t:i for i,t in enumerate(config.STOCKS)}
        src, dst = [], []
        for _, row in edf.iterrows():
            s, d = row["source"], row["target"]
            if s in idx_map and d in idx_map:
                src.append(idx_map[s]); dst.append(idx_map[d])
        if src:
            edge_index = torch.tensor([src+dst, dst+src], dtype=torch.long)
        else:
            edge_index = torch.empty((2,0),dtype=torch.long)
    else:
        edge_index = torch.empty((2,0),dtype=torch.long)

    # --- assemble per‐day graphs ---
    dates      = sorted(next(iter(feats.values())).index)
    train_list, test_list = [], []

    for d in dates:
        x_rows, y_rows = [], []
        feat_len = next(iter(feats.values())).shape[1]
        for t in config.STOCKS:
            df = feats.get(t)
            if df is None or d not in df.index:
                x_rows.append(np.zeros(feat_len))
                y_rows.append(0)
            else:
                x_rows.append(df.loc[d].values)
                y_rows.append(int(tgts[t].loc[d]))
        data = Data(
            x=torch.tensor(np.vstack(x_rows), dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y_rows, dtype=torch.long)
        )
        if pd.to_datetime(d) < pd.to_datetime(split_date):
            train_list.append(data)
        else:
            test_list.append(data)

    print(f"Built {len(train_list)} train / {len(test_list)} test graphs")
    return train_list, test_list
