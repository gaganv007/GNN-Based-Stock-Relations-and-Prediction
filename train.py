import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models import GCN, GAT, GraphSAGE, TemporalGNN
from utils import save_model, calculate_metrics
from sklearn.ensemble import RandomForestClassifier
import config

MODEL_MAP = {
    "GCN":         GCN,
    "GAT":         GAT,
    "GraphSAGE":   GraphSAGE,
    "TemporalGNN": TemporalGNN,
    # no LSTM
}

def train_model(name, train_data, test_data):
    if name == "RandomForest":
        X = np.vstack([d.x.numpy() for d in train_data])
        y = np.hstack([d.y.numpy() for d in train_data])
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        save_model(rf, name)
        print(f"✓ Saved RandomForest (n_estimators=100)")
        return rf, None

    ModelClass = MODEL_MAP[name]
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = ModelClass(train_data[0].x.shape[1]).to(device)
    optimiser  = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loader     = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    best_acc = 0.0
    for epoch in range(1, config.EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out   = model(batch.x, batch.edge_index)
            loss  = F.cross_entropy(out, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        # evaluate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for d in test_data:
                d = d.to(device)
                o = model(d.x, d.edge_index)
                ys.append(d.y.cpu().numpy())
                ps.append(F.softmax(o, dim=1).cpu().numpy())
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = y_prob.argmax(axis=1)
        m = calculate_metrics(y_true, y_pred, y_prob)

        print(f"[{name}] Epoch {epoch} — loss {total_loss/len(loader):.4f}  "
              f"val acc {m['accuracy']:.3f}  f1 {m['f1']:.3f}")

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            save_model(model, name)

    print(f"✓ Saved best {name}.pt  (acc={best_acc:.3f})")
    return model, None
