import os
import json
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from utils import load_model
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph

def evaluate_gnn(model_name: str, batch_size: int = config.BATCH_SIZE):
    feats, targets = FeatureEngineer({}).generate_features()
    pred = config.STOCKS[0]
    dates = feats[pred].index

    gc = construct_graph(features=feats)
    G  = gc.build_combined_graph()
    node2idx = {tk: i for i, tk in enumerate(config.STOCKS)}

    data_list, data_dates, y_true = [], [], []
    for i, date in enumerate(dates[:-1]):
        y_true.append(int(targets[pred].loc[dates[i+1]]))

        x = torch.tensor(
            [feats[tk].loc[date].values for tk in config.STOCKS],
            dtype=torch.float
        )
        edges = []
        for u, v in G.edges():
            ui, vi = node2idx[u], node2idx[v]
            edges += [(ui, vi), (vi, ui)]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()

        d = Data(x=x, edge_index=ei, y=torch.tensor([y_true[-1]], dtype=torch.long))
        data_list.append(d)
        data_dates.append(date)

    mask = [d >= pd.to_datetime(config.TEST_START_DATE) for d in data_dates]
    test_data = [d for d, m in zip(data_list, mask) if m]
    y_test    = [y for y, m in zip(y_true, mask)     if m]

    loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load best model
    path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{model_name}.pt not found")
    model = load_model(model_name)
    model.eval()

    y_pred = []
    with torch.no_grad():
        for batch in loader:
            out  = model(batch)
            y_pred += out.argmax(dim=1).cpu().tolist()

    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0)
    }

def evaluate_all():
    results = {}
    for m in ["GCN","GAT","GraphSAGE","TemporalGNN"]:
        path = os.path.join(config.MODELS_DIR, f"{m}.pt")
        if os.path.exists(path):
            print(f"Evaluating {m}…")
            results[m] = evaluate_gnn(m)
        else:
            print(f"Skipping {m}: no weights.")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    print("Test‐period metrics:", evaluate_all())
