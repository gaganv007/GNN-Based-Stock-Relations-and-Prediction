import os
import json
import config
import pandas as pd

from train import train_rf, train_lstm
from feature_engineering import FeatureEngineer

def evaluate_all():
    """
    Generate and save evaluation metrics for RandomForest and LSTM models.
    """
    # 1. Generate features and targets
    feats, targets = FeatureEngineer().generate_features()

    # 2. Train and evaluate each model
    rf_metrics   = train_rf(feats, targets)
    lstm_metrics = train_lstm(feats, targets)

    # 3. Aggregate metrics
    all_metrics = {
        'RandomForest': rf_metrics,
        'LSTM':        lstm_metrics
    }

    # 4. Save metrics to disk
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(config.RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

if __name__ == '__main__':
    results = evaluate_all()
    print('Evaluation complete. Metrics:')
    print(results)
