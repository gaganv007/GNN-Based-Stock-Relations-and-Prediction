import argparse
from data_collection import download_all_data
from feature_engineering import FeatureEngineer
from graph_construction import construct_graph
from train import train_gnn
from evaluate import evaluate_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true", help="Run download → features → graph → train & eval all models")
    parser.add_argument("--model", type=str,
                        choices=["GCN","GAT","GraphSAGE","TemporalGNN"],
                        help="Train & evaluate only this model")
    args = parser.parse_args()

    if args.all:
        print("1) Downloading data…");     download_all_data()
        print("2) Generating features…");  FeatureEngineer({}).generate_features()
        print("3) Constructing graph…");    construct_graph(features=FeatureEngineer({}).features)
        print("4) Training all GNN models…")
        for m in ["GCN","GAT","GraphSAGE","TemporalGNN"]:
            print(f"   └─ Training {m}"); train_gnn(m)
    elif args.model:
        print("1) Downloading data…");     download_all_data()
        print("2) Generating features…");  FeatureEngineer({}).generate_features()
        print("3) Constructing graph…");    construct_graph(features=FeatureEngineer({}).features)
        print(f"4) Training {args.model}…"); train_gnn(args.model)
    else:
        parser.print_help(); exit(1)

    print("5) Evaluating GNN models…")
    results = evaluate_all()
    print("Metrics:", results)
