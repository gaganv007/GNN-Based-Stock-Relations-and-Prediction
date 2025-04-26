# train.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from models import get_model
from utils import calculate_metrics, save_model
import config

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1-pt)**gamma * ce).mean()

def train_model(name, train_data, test_data):
    if name == "RandomForest":
        X = np.vstack([d.x.numpy() for d in train_data])
        y = np.hstack([d.y.numpy() for d in train_data])
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        from joblib import dump
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        dump(rf, os.path.join(config.MODELS_DIR, f"{name}.joblib"))
        print(f"✓ Trained & saved RandomForest.joblib")
        return

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = get_model(name, input_dim=train_data[0].x.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=5)
    loader    = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    best_acc = 0.0
    no_imp   = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out = model(batch.x, batch.edge_index)
            logits = out[0] if isinstance(out, tuple) else out
            loss = focal_loss(logits, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        ys, ps, preds = [], [], []
        with torch.no_grad():
            for d in test_data:
                d = d.to(device)
                out = model(d.x, d.edge_index)
                logits = out[0] if isinstance(out, tuple) else out
                prob   = F.softmax(logits, dim=1).cpu().numpy()
                pred   = prob.argmax(axis=1)
                ys.append(d.y.cpu().numpy())
                ps.append(prob)
                preds.append(pred)

        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = np.concatenate(preds)
        m = calculate_metrics(y_true, y_pred, y_prob)

        print(f"[{name}] Epoch {epoch}  loss={total_loss/len(loader):.4f}  val_acc={m['accuracy']:.3f}")
        scheduler.step(m["accuracy"])

        if m["accuracy"] > best_acc + 1e-4:
            best_acc = m["accuracy"]
            no_imp   = 0
            save_model(model, name)
        else:
            no_imp += 1
            if no_imp >= 10:
                print("Early stopping")
                break

    print(f"✓ Best {name}.pt saved (acc={best_acc:.3f})")
