import os
import argparse
import json
from datetime import datetime

import config
from data_collection import download_price_data as download_all_data
from feature_engineering import prepare_features
from graph_construction import construct_graph
from train import train_model
from evaluate import evaluate_all_models
from utils import load_train_test_data


def parse_args():
    parser = argparse.ArgumentParser(description="GNN-based stock prediction")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--prepare", action="store_true", help="Prepare features")
    parser.add_argument("--construct", action="store_true", help="Construct graph")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--model", type=str, default=None, help="Specific model to train/evaluate")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1) Download
    if args.download or args.all:
        print("1) Downloading data…")
        download_all_data(
            stocks=config.STOCKS,
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )

    # 2) Feature preparation
    if args.prepare or args.all:
        print("2) Preparing features…")
        prepare_features()

    # 3) Graph construction
    if args.construct or args.all:
        print("3) Constructing graph…")
        construct_graph()

    # 4) Training
    if args.train or args.all:
        print("4) Training model(s)…")
        train_model(model_name=args.model, tune=args.tune)

    # 5) Evaluation
    if args.evaluate or args.all:
        print("5) Evaluating model(s)…")
        evaluate_all_models(model_name=args.model)


if __name__ == "__main__":
    main()
