from data_collection    import download_all_data
from feature_engineering import prepare_features
from graph_construction import construct_graph
from utils              import load_train_test_data
from train              import train_model
import config

def main():
    # 1) Download raw data
    download_all_data()

    # 2) Prepare features & targets
    feats, tgts = prepare_features()

    # 3) Construct graph
    _ = construct_graph(feats)

    # 4) Build train/test PyG datasets
    train_list, test_list = load_train_test_data(feats, tgts, split_date=config.TEST_START_DATE)

    # 5) Train every model
    for name in config.MODELS:
        print(f"\n└─ Training {name}")
        train_model(name, train_list, test_list)

if __name__ == "__main__":
    main()
