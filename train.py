import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import config
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from models import get_model

def train_gnn(model_name: str,
              lr: float   = 0.001,
              epochs: int = 50,
              batch_size: int = 8):
    """
    Train a GNN to predict next-day up/down for the first stock in config.STOCKS.
    """

    # 1) Generate features & targets
    feats, targets = FeatureEngineer().generate_features()
    pred_ticker = config.STOCKS[0]
    dates = feats[pred_ticker].index

    # 2) Stack all features to compute mean/std for normalization
    all_feat_arrays = []
    for df in feats.values():
        all_feat_arrays.append(df.values)
    all_feats = np.vstack(all_feat_arrays)  # [num_dates * num_stocks, num_features]
    feat_mean = all_feats.mean(axis=0)
    feat_std  = all_feats.std(axis=0) + 1e-6

    # 3) Build combined graph and integer edge_index
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

    # 4) Create one Data object per date
    data_list = []
    for i in range(len(dates) - 1):
        date = dates[i]
        # Node features at this date, normalized
        x_np = np.vstack([feats[stk].loc[date].values for stk in config.STOCKS])
        x_np = (x_np - feat_mean) / feat_std
        x = torch.tensor(x_np, dtype=torch.float)

        # Label = next-day movement of pred_ticker
        label = int(targets[pred_ticker].loc[dates[i+1]])
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=True)

    # 5) Initialize model, optimizer, loss
    model = get_model(model_name, config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 6) Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[Train] {model_name} Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

    # 7) Save trained model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"{model_name}.pt"))
    print(f"Saved {model_name}.pt")
    return model
