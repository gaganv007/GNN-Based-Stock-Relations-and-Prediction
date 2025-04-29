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
import matplotlib.pyplot as plt

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()

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

    losses, accs, precs, recs, f1s, aucs = [], [], [], [], [], []
    best_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out    = model(batch.x, batch.edge_index)
            logits = out[0] if isinstance(out, tuple) else out
            loss   = focal_loss(logits, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # —— VALIDATION —— 
        model.eval()
        ys, ps, preds = [], [], []
        with torch.no_grad():
            for d in test_data:
                d      = d.to(device)
                out    = model(d.x, d.edge_index)
                logits = out[0] if isinstance(out, tuple) else out
                prob   = F.softmax(logits, dim=1).cpu().numpy()
                pred   = prob.argmax(axis=1)
                ys.append(d.y.cpu().numpy())
                ps.append(prob)
                preds.append(pred)

        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = np.concatenate(preds)
        m      = calculate_metrics(y_true, y_pred, y_prob)
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])
        aucs.append(m["roc_auc"])

        print(
            f"[{name}] Epoch {epoch:2d}  "
            f"loss={avg_loss:.4f}  "
            f"acc={m['accuracy']:.3f}  "
            f"prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  "
            f"auc={m['roc_auc']:.3f}"
        )
        scheduler.step(m["accuracy"])

        if m["accuracy"] > best_acc + 1e-4:
            best_acc = m["accuracy"]
            save_model(model, name)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metric_series = {
        "loss":      losses,
        "accuracy":  accs,
        "precision": precs,
        "recall":    recs,
        "f1":        f1s,
        "roc_auc":   aucs,
    }
    for metric_name, vals in metric_series.items():
        plt.figure()
        plt.plot(range(1, len(vals)+1), vals)
        plt.title(f"{name} {metric_name} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(
            config.RESULTS_DIR, f"{name}_{metric_name}.png"
        ))
        plt.close()

    print(f"✓ Training complete. Best acc={best_acc:.3f}")
    print(f"✓ All metric curves saved to {config.RESULTS_DIR}")
