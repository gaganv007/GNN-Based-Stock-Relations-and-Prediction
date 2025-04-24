import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from models import get_model
from utils import calculate_metrics
import config

def train_model(name, train_data, test_data, epochs=30, lr=1e-3, batch_size=16):
    if name == "RandomForest":
        X = np.vstack([d.x.numpy() for d in train_data])
        y = np.hstack([d.y.numpy() for d in train_data])

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)

        from joblib import dump
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        dump(rf, os.path.join(config.MODELS_DIR, f"{name}.joblib"))
        print(f"✓ Trained & saved RandomForest as {name}.joblib")
        return rf, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = get_model(name, input_dim=train_data[0].x.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    loader    = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

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

        print(f"[{name}] Epoch {epoch}: loss {total_loss/len(loader):.4f}  val_acc {m['accuracy']:.3f}")

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            from utils import save_model
            save_model(model, name)

    print(f"✓ Saved best {name}.pt  (acc={best_acc:.3f})")
    return model, None
