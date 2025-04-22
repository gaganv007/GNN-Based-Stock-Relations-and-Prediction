import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from utils import load_model
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph

def evaluate_gnn(model_name: str, batch_size: int = 8):
    """
    Evaluate a single GNN on next‐day up/down classification.
    """
    feats, targets = FeatureEngineer().generate_features()
    pred_ticker = config.STOCKS[0]
    dates = feats[pred_ticker].index

    # Build normalization stats (must match train)
    all_feat_arrays = [df.values for df in feats.values()]
    all_feats = np.vstack(all_feat_arrays)
    feat_mean = all_feats.mean(axis=0)
    feat_std  = all_feats.std(axis=0) + 1e-6

    # Build edge_index
    gc = construct_graph()
    G = gc.build_combined_graph()
    node2idx = {stk: idx for idx, stk in enumerate(config.STOCKS)}
    edge_list = []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            ui, vi = node2idx[u], node2idx[v]
            edge_list.append((ui, vi))
            edge_list.append((vi, ui))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Build evaluation dataset
    data_list, y_true = [], []
    for i in range(len(dates) - 1):
        date = dates[i]
        x_np = np.vstack([feats[stk].loc[date].values for stk in config.STOCKS])
        x_np = (x_np - feat_mean) / feat_std
        x = torch.tensor(x_np, dtype=torch.float)

        label = int(targets[pred_ticker].loc[dates[i + 1]])
        y_true.append(label)

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        data_list.append(data)

    loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved weights for {model_name} at {model_path}")

    model = load_model(model_name)
    model.eval()

    # Inference
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            preds = out.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)

    # Compute metrics
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate_all():
    """
    Evaluate each GNN variant for which a .pt file exists,
    save results to metrics.json, and return the dict.
    """
    all_results = {}
    for m in ["GCN","GAT","GraphSAGE","TemporalGNN"]:
        model_path = os.path.join(config.MODELS_DIR, f"{m}.pt")
        if not os.path.exists(model_path):
            print(f"  • Skipping {m}: no weights at {model_path}")
            continue
        print(f"  • Evaluating {m}…")
        all_results[m] = evaluate_gnn(m)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results

if __name__ == "__main__":
    results = evaluate_all()
    print("Evaluation complete. Metrics:", results)
