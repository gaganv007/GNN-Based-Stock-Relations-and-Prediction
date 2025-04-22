import argparse
from data_collection import download_all_data
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from train import train_rf, train_lstm
from evaluate import evaluate_all

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        # 1. Download & merge
        download_all_data()

        # 2. Feature engineering
        feats, targets = FeatureEngineer().generate_features()

        # 3. Graph construction
        construct_graph()

        # 4. Train models
        rf_metrics   = train_rf(feats, targets)
        lstm_metrics = train_lstm(feats, targets)

        # 5. Evaluate
        all_metrics = evaluate_all()
        print("Done. Metrics:", all_metrics)
