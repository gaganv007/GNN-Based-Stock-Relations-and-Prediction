# main.py

import os
import argparse
import pandas as pd
import config

from data_collection import download_all_data
from feature_engineering import prepare_features
from graph_construction import construct_graph
from train import train_model
from evaluate import evaluate_all_models
from utils import load_train_test_data

def parse_args():
    p = argparse.ArgumentParser("GNN Stock Pipeline")
    p.add_argument("--download",  action="store_true")
    p.add_argument("--prepare",   action="store_true")
    p.add_argument("--construct", action="store_true")
    p.add_argument("--train",     action="store_true")
    p.add_argument("--evaluate",  action="store_true")
    p.add_argument("--all",       action="store_true")
    p.add_argument("--model",     type=str, default=None)
    p.add_argument("--tune",      action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    # ensure dirs
    for d in (config.RAW_DIR, config.PROCESSED_DIR,
              config.GRAPHS_DIR, config.MODELS_DIR, config.RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    # 1) Download raw CSV
    if args.download or args.all:
        print("1) Downloading data…")
        raw_df = download_all_data()

    # 2) Prepare features/targets
    if args.prepare or args.all:
        print("2) Preparing features…")
        feats, targets = prepare_features(
            raw_csv=os.path.join(config.RAW_DIR, "stock_data.csv"),
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )

    # 3) Build graphs
    if args.construct or args.all:
        print("3) Constructing graph…")
        graph_data = construct_graph(feats)

    # 4) Train
    if args.train or args.all:
        print("4) Training models…")
        train_data, test_data = load_train_test_data(feats, targets)
        models = [args.model] if args.model else config.MODELS
        for m in models:
            print(f"→ {m}")
            train_model(m, train_data, test_data, tune=args.tune)

    # 5) Evaluate
    if args.evaluate or args.all:
        print("5) Evaluating…")
        evaluate_all_models(
            [args.model] if args.model else config.MODELS,
            feats, targets
        )

if __name__ == "__main__":
    main()
