# main.py
from data_collection import DataCollector
from feature_engineering import FeatureEngineer
from graph_construction import build_static_graph
import train

def main():
    # 1) Download & cache
    dc = DataCollector()
    dc.download_price_data()

    # 2) Load raw
    raw = dc.load_all()

    # 3) Features & targets
    fe = FeatureEngineer(raw)
    feats, targets, dates = fe.make_features()

    # 4) Build graph
    #   use Close‐prices of last available date for correlation
    last_prices = {t: raw[t]["Close"].loc[dates[t]] for t in feats}
    G = build_static_graph(last_prices)

    # 5) Train & evaluate
    train.train_and_eval(feats, targets, G, dates)

if __name__=="__main__":
    main()
