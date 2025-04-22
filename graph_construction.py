import os
import pickle
import pandas as pd
import networkx as nx
import config
from feature_engineering import FeatureEngineer

class GraphConstructor:
    def __init__(self, features):
        """
        features: dict of ticker -> DataFrame of engineered features
        """
        self.features = features

    def build_correlation_graph(self):
        rets = {stk: feat["returns_1d"] for stk, feat in self.features.items()}
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
        for grp in config.SECTOR_MAPPING.values():
            for a in grp:
                for b in grp:
                    if a != b:
                        G.add_edge(a, b, relation="sector")
        return G

    def build_combined_graph(self):
        G1 = self.build_correlation_graph()
        G2 = self.build_sector_graph()
        return nx.compose(G1, G2)

def construct_graph(features=None):
    """
    Return a GraphConstructor built on `features`.
    If `features` is None, it falls back to generating them via FeatureEngineer.
    """
    if features is None:
        feats, _ = FeatureEngineer().generate_features()
    else:
        feats = features
    return GraphConstructor(feats)
