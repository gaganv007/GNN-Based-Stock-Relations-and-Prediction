import networkx as nx
import pandas as pd
import numpy as np
from config import CORRELATION_THRESHOLD, SECTOR_MAPPING

def build_correlation_graph(feature_dict):
    """
    Build a static undirected graph where edge (i,j) exists if
    correlation( returns_1d_i, returns_1d_j ) ≥ threshold.
    """
    rets = {t: f["returns_1d"].values for t, f in feature_dict.items()}
    corr = pd.DataFrame(rets).corr()
    G = nx.Graph()
    G.add_nodes_from(feature_dict.keys())
    for i in corr.index:
        for j in corr.columns:
            if i>=j: continue
            if corr.at[i,j] >= CORRELATION_THRESHOLD:
                G.add_edge(i,j, weight=corr.at[i,j])
    print(f"  • Correlation edges: {G.number_of_edges()}")
    return G

def add_sector_edges(G):
    """
    Add edges between stocks in the same sector.
    """
    for sector, tickers in SECTOR_MAPPING.items():
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                G.add_edge(tickers[i], tickers[j], weight=1.0)
    print(f"  • After sector edges: {G.number_of_edges()}")
    return G
