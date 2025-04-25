# main.py

from data_collection     import download_all_data
from feature_engineering import prepare_features
from graph_construction  import construct_graph
from utils               import load_train_test_data
from train               import train_model
import config

def main():
    # 1) download & prepare data
    download_all_data()
    feats, tgts = prepare_features()
    construct_graph(feats)

    # 2) split into train/test
    train_list, test_list = load_train_test_data(feats, tgts, split_date=config.TEST_START_DATE)

    # 3) train each model
    for name in config.MODELS:
        print(f"\n└─ Training {name}")
        train_model(name, train_list, test_list)

if __name__ == "__main__":
    main()
