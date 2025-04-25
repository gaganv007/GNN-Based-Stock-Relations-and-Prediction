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

def train_model(name, train_data, test_data):
    # --- RandomForest branch unchanged ---
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

    # --- GNN branch ---
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = get_model(name, input_dim=train_data[0].x.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=config.LR)
    loader    = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    best_acc = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()

            # Call the model
            out = model(batch.x, batch.edge_index)

            # Unpack logits depending on model
            if isinstance(out, tuple):
                logits, _ = out      # GATWithAtt returns (logits, attn)
            else:
                logits = out         # GCN/GraphSAGE return just logits

            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        ys, ps, preds = [], [], []
        with torch.no_grad():
            for d in test_data:
                d_out = model(d.x.to(device), d.edge_index.to(device))
                if isinstance(d_out, tuple):
                    logits, _ = d_out
                else:
                    logits = d_out

                prob = F.softmax(logits, dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                ys.append(d.y.cpu().numpy())
                ps.append(prob)
                preds.append(pred)

        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = np.concatenate(preds)
        m = calculate_metrics(y_true, y_pred, y_prob)

        print(f"[{name}] Epoch {epoch}  loss={total_loss/len(loader):.4f}  val_acc={m['accuracy']:.3f}")

        # Save best
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            save_model(model, name)

    print(f"✓ Best {name}.pt saved (acc={best_acc:.3f})")
