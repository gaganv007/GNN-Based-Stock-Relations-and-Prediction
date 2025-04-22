import networkx as nx
import pandas as pd
import config

class GraphConstructor:
    def __init__(self, features):
        """
        features: dict of ticker -> feature DataFrame
        """
        self.features = features

    def build_correlation_graph(self):
        rets = {tk: feat["returns_1d"] for tk, feat in self.features.items()}
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
    """
    if features is None:
        from feature_engineering import FeatureEngineer
        feats, _ = FeatureEngineer({}).generate_features()
    else:
        feats = features
    return GraphConstructor(feats)
