import os
import torch
import argparse
import json
from datetime import datetime
import config

# this now works, because we've defined it above
from data_collection import download_all_data
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from train import train_model         # make sure train.py exports train_model
from evaluate import evaluate_all_models  # and evaluate.py exports evaluate_all_models
from utils import load_train_test_data

def parse_args():
    parser = argparse.ArgumentParser(description='GNN-based stock prediction')
    parser.add_argument('--download', action='store_true', help='Download data')
    parser.add_argument('--prepare', action='store_true', help='Prepare features')
    parser.add_argument('--construct', action='store_true', help='Construct graph')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--model', type=str, default=None, help='Specific model to train/evaluate')
    return parser.parse_args()

def main():
    args = parse_args()

    # step 1: download
    if args.download or args.all:
        print("1) Downloading data…")
        data = download_all_data()

    # step 2: feature prep
    if args.prepare or args.all:
        print("2) Generating features…")
        feats, targets = FeatureEngineer(data).generate_features()

    # step 3: graph
    if args.construct or args.all:
        print("3) Constructing graph…")
        graph_data = construct_graph(feats)

    # step 4: train
    if args.train or args.all:
        print("4) Training models…")
        train_model(args.model)

    # step 5: eval
    if args.evaluate or args.all:
        print("5) Evaluating models…")
        results = evaluate_all_models()
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
import os
import torch; torch.manual_seed(42)
from data_collection import download_price_data
from feature_engineering import compute_technical_features
from graph_construction import build_correlation_graph, add_sector_edges
from evaluate import full_evaluate

if __name__=="__main__":
    os.makedirs("saved_models", exist_ok=True)

    # 1) Data
    prices = download_price_data()

    # 2) Features
    print("\n▶︎ Computing technical features …")
    feat = compute_technical_features(prices)

    # 3) Graph
    print("\n▶︎ Building correlation graph …")
    G = build_correlation_graph(feat)
    print("▶︎ Adding sector edges …")
    G = add_sector_edges(G)

    # 4) Train & evaluate
    print("\n▶︎ Train & evaluate pipeline …")
    full_evaluate(prices, feat, G)
