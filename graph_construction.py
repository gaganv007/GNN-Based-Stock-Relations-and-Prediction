import os
import pickle
import pandas as pd          # <— added this import
import networkx as nx
import config
from feature_engineering import FeatureEngineer

class GraphConstructor:
    def __init__(self, features):
        self.features = features

    def build_correlation_graph(self):
        # Build a DataFrame of 1‑day returns and compute correlations
        rets = {stk: feat['returns_1d'] for stk, feat in self.features.items()}
        corr = pd.DataFrame(rets).corr()
        G = nx.Graph()
        G.add_nodes_from(self.features.keys())
        for i in corr.index:
            for j in corr.columns:
                if i != j and corr.loc[i, j] > config.CORRELATION_THRESHOLD:
                    G.add_edge(i, j, weight=float(corr.loc[i, j]))
        return G

    def build_sector_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.features.keys())
        for group in config.SECTOR_MAPPING.values():
            for a in group:
                for b in group:
                    if a != b:
                        G.add_edge(a, b, relation='sector')
        return G

    def build_combined_graph(self):
        G1 = self.build_correlation_graph()
        G2 = self.build_sector_graph()
        return nx.compose(G1, G2)

    def save_graphs(self):
        os.makedirs(os.path.join(config.DATA_DIR, 'graphs'), exist_ok=True)
        graphs = [
            ('corr',     self.build_correlation_graph()),
            ('sector',   self.build_sector_graph()),
            ('combined', self.build_combined_graph())
        ]
        for name, G in graphs:
            path = os.path.join(config.DATA_DIR, 'graphs', f"{name}_graph.pkl")
            with open(path, 'wb') as f:
                pickle.dump(G, f)

def construct_graph():
    feats, _ = FeatureEngineer().generate_features()
    gc = GraphConstructor(feats)
    gc.save_graphs()
    return gc
