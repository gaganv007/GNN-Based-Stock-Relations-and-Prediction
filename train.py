# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import config
from utils import save_model, calculate_metrics
from models import GCN, GAT, GraphSAGE, TemporalGNN

MODEL_MAP = {
    "GCN":    GCN,
    "GAT":    GAT,
    "GraphSAGE": GraphSAGE,
    "TemporalGNN": TemporalGNN
}

def train_model(name, train_data, test_data, tune=False):
    """
    Train a single GNN model `name` on train_data, evaluate on test_data.
    Saves best weights to MODELS_DIR.
    """
    print(f"   └─ Training {name}")
    Model = MODEL_MAP[name]
    in_dim = train_data[0].x.shape[1]
    net = Model(in_dim).to(config.DEVICE)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loader = DataLoader(train_data, batch_size=16, shuffle=True)
    best_acc = 0.0

    for epoch in range(1, 31):
        net.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(config.DEVICE)
            opt.zero_grad()
            out = net(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[{name}] Epoch {epoch} — loss {total_loss/len(loader):.4f}")

        # eval
        net.eval()
        ys, ps = [], []
        for d in test_data:
            d = d.to(config.DEVICE)
            with torch.no_grad():
                out = net(d.x, d.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()
            ys.append(d.y.cpu().numpy())
            ps.append(F.softmax(out, dim=1).cpu().numpy())
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = y_prob.argmax(axis=1)
        m = calculate_metrics(y_true, y_pred, y_prob)
        print(f"   → val acc {m['accuracy']:.3f}  f1 {m['f1']:.3f}")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            save_model(net, name)
    print(f"   ✓ Saved best {name}.pt  (acc={best_acc:.3f})")
