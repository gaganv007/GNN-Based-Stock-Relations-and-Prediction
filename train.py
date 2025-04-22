import os
import pandas as pd
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
              lr: float   = config.LEARNING_RATE,
              epochs: int = config.NUM_EPOCHS,
              batch_size: int = config.BATCH_SIZE):
    feats, targets = FeatureEngineer({}).generate_features()
    pred = config.STOCKS[0]
    dates = feats[pred].index

    # Build graph once
    gc = construct_graph(features=feats)
    G  = gc.build_combined_graph()
    node2idx = {tk: i for i, tk in enumerate(config.STOCKS)}

    # Prepare data objects + record dates
    data_list, data_dates = [], []
    for i, date in enumerate(dates[:-1]):
        x = torch.tensor(
            [feats[tk].loc[date].values for tk in config.STOCKS],
            dtype=torch.float
        )
        y = torch.tensor([int(targets[pred].loc[dates[i+1]])], dtype=torch.long)

        edges = []
        for u, v in G.edges():
            ui, vi = node2idx[u], node2idx[v]
            edges += [(ui, vi), (vi, ui)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        d = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(d)
        data_dates.append(date)

    # Train split
    train_mask = [d <= pd.to_datetime(config.TRAIN_END_DATE) for d in data_dates]
    train_data = [d for d, m in zip(data_list, train_mask) if m]

    loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)

    model     = get_model(model_name, config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    model.train()
    best_loss = float("inf")
    patience  = 0

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"[Train] {model_name} Epoch {epoch}/{epochs} — Loss: {avg:.4f}")

        # Early stopping
        if avg < best_loss:
            best_loss, patience = avg, 0
            # Save best
            torch.save(model.state_dict(),
                       os.path.join(config.MODELS_DIR, f"{model_name}.pt"))
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

    print(f"Final weights saved for {model_name}.pt")
    return model
