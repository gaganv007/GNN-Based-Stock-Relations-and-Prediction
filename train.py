import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score
from tqdm import trange
from config import DEVICE, LR, EPOCHS, PATIENCE, BATCH_SIZE, HIDDEN_DIM, DROPOUT
from models import GCNNet

def prepare_dataset(feature_dict, G):
    """
    Build a single PyG Data object:
      - x:   [num_nodes×num_feats]
      - y:   [num_nodes]
      - edge_index: 2×E long tensor
    Node order = sorted(feature_dict.keys())
    """
    nodes = sorted(feature_dict)
    feats = [feature_dict[n].iloc[-1].drop("target").values for n in nodes]
    y     = [feature_dict[n].iloc[-1]["target"] for n in nodes]
    x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y,       dtype=torch.long,    device=DEVICE)
    # edges
    idx = [nodes.index(u) for u,v in G.edges()]
    jdx = [nodes.index(v) for u,v in G.edges()]
    edge_index = torch.tensor([idx+jdx, jdx+idx], dtype=torch.long, device=DEVICE)
    data = Data(x=x, y=y, edge_index=edge_index)
    return data

def train_gcn(data_train, data_val, in_dim, out_dim):
    model = GCNNet(in_dim, HIDDEN_DIM, out_dim, DROPOUT).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc, patience = 0.0, 0
    for epoch in trange(1, EPOCHS+1, desc="Training GCN"):
        model.train()
        opt.zero_grad()
        out = model(data_train.x, data_train.edge_index)
        loss = cross_entropy(out, data_train.y)
        loss.backward()
        opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            pred = model(data_val.x, data_val.edge_index).argmax(dim=1)
            val_acc = accuracy_score(data_val.y.cpu(), pred.cpu())

        print(f"  Epoch {epoch:02d}: loss={loss.item():.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc; torch.save(model.state_dict(), "saved_models/gcn.pt"); patience=0
        else:
            patience +=1
            if patience>=PATIENCE:
                print("  → Early stopping")
                break
    return best_val_acc

def load_gcn(in_dim, out_dim):
    model = GCNNet(in_dim, HIDDEN_DIM, out_dim, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load("saved_models/gcn.pt"))
    model.eval()
    return model
