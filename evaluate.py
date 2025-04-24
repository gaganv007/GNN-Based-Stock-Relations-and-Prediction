# evaluate.py

import torch
import numpy as np
import config
from utils import load_model, load_train_test_data, calculate_metrics

def evaluate_all_models(names, feats, targets):
    """
    For each model in names, load its weights, evaluate on test set,
    and print a summary table.
    """
    train_data, test_data = load_train_test_data(feats, targets)
    print("\n=== Evaluation ===")
    for name in names:
        print(f"→ {name}")
        net = load_model(name, in_dim=train_data[0].x.shape[1])
        net.eval()
        ys, ps = [], []
        for d in test_data:
            d = d.to(config.DEVICE)
            with torch.no_grad():
                out = net(d.x, d.edge_index)
            ys.append(d.y.cpu().numpy())
            ps.append(torch.nn.functional.softmax(out, dim=1).cpu().numpy())
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = y_prob.argmax(axis=1)
        m = calculate_metrics(y_true, y_pred, y_prob)
        print(f"   acc={m['accuracy']:.3f}  precision={m['precision']:.3f}"
              f"  recall={m['recall']:.3f}  f1={m['f1']:.3f}  roc={m['roc_auc']:.3f}")
