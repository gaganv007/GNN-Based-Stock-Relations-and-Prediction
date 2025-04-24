import os
import argparse
from dotenv import load_dotenv

import config
from data_collection import DataCollector
from feature_engineering import prepare_features
from graph_construction import construct_graph
from train import train_model
from evaluate import evaluate_all_models
from utils import load_train_test_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN-based stock prediction pipeline"
    )
    parser.add_argument('--download',   action='store_true', help="Download raw data")
    parser.add_argument('--prepare',    action='store_true', help="Prepare features")
    parser.add_argument('--construct',  action='store_true', help="Construct graph")
    parser.add_argument('--train',      action='store_true', help="Train models")
    parser.add_argument('--evaluate',   action='store_true', help="Evaluate models")
    parser.add_argument('--all',        action='store_true', help="Run all steps")
    parser.add_argument('--model',      type=str,       help="Specific model to train/evaluate")
    parser.add_argument('--tune',       action='store_true', help="Tune hyperparameters")
    return parser.parse_args()

def main():
    load_dotenv()  # loads NEWS_API_KEY, FRED_API_KEY, etc.
    args = parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1) Download
    if args.download or args.all:
        print("1) Downloading raw data…")
        dc = DataCollector()
        dc.download_price_data()
        dc.download_news_data()
        dc.download_fred_data(getattr(config, "FRED_SERIES_IDS", []))

    # 2) Prepare features
    if args.prepare or args.all:
        print("\n2) Preparing features…")
        prepare_features()

    # 3) Construct graph
    if args.construct or args.all:
        print("\n3) Constructing graph…")
        construct_graph()

    # 4) Train
    if args.train or args.all:
        print("\n4) Training models…")
        train_data, test_data = load_train_test_data()
        if args.model:
            print(f"  • Training {args.model}…")
            if args.tune:
                from train import Trainer
                trainer = Trainer(args.model)
                best = trainer.tune_hyperparameters(train_data, test_data)
                train_model(args.model, train_data, test_data, **best)
            else:
                train_model(args.model, train_data, test_data)
        else:
            for m in config.MODELS:
                print(f"  • Training {m}…")
                if args.tune:
                    from train import Trainer
                    trainer = Trainer(m)
                    best = trainer.tune_hyperparameters(train_data, test_data)
                    train_model(m, train_data, test_data, **best)
                else:
                    train_model(m, train_data, test_data)

    # 5) Evaluate
    if args.evaluate or args.all:
        print("\n5) Evaluating models…")
        if args.model:
            evaluate_all_models([args.model])
        else:
            evaluate_all_models()

    print("\nDone.")

if __name__ == "__main__":
    main()