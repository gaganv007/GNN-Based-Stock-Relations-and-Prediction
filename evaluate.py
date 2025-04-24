import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils import load_train_test_data, calculate_metrics, load_model
import config

def evaluate_all_models(model_list, features_pkl, targets_pkl, graph_dir, device=None):
    """
    Load feature/target pickles and edge‐list, then evaluate each model.
    Returns a dict of metrics for each model.
    """
    with open(features_pkl, "rb") as f:
        feats = pickle.load(f)
    with open(targets_pkl, "rb") as f:
        tgts = pickle.load(f)

    train_data, test_data = load_train_test_data(feats, tgts, split_date=config.TEST_START_DATE)
    results = {}

    for name in model_list:
        print(f"Evaluating {name} …")
        if name == "RandomForest":
            import joblib
            rf = joblib.load(os.path.join(config.MODELS_DIR, f"{name}.joblib"))
            X = np.vstack([d.x.cpu().numpy() for d in test_data])
            y = np.hstack([d.y.cpu().numpy() for d in test_data])
            prob = rf.predict_proba(X)
            pred = prob.argmax(axis=1)
            m    = calculate_metrics(y, pred, prob)
        else:
            mdl = load_model(name, in_dim=test_data[0].x.shape[1], device=device)
            from torch_geometric.loader import DataLoader
            y_true, y_pred, y_prob = [], [], []
            for batch in DataLoader(test_data, batch_size=config.BATCH_SIZE):
                batch = batch.to(device)
                out   = mdl(batch.x, batch.edge_index)
                p     = F.softmax(out, dim=1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(p.argmax(axis=1).tolist())
                y_prob.extend(p.tolist())
            m = calculate_metrics(y_true, y_pred, np.array(y_prob))
        results[name] = m

    return results
