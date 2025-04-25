# graph_construction.py

import os
import pandas as pd
import networkx as nx
import config
from datetime import timedelta

def dynamic_correlation_graph(feats: dict, window: int, threshold: float):
    # Use most recent `window` days to compute correlation
    # Find common dates
    common = set.intersection(*(set(df.index) for df in feats.values()))
    end_date = max(common)
    start_date = end_date - timedelta(days=window)
    # Build DataFrame of returns for that window
    ret_map = {
        t: feats[t]['returns_1d'].loc[start_date:end_date]
        for t in feats
        if 'returns_1d' in feats[t]
    }
    ret_df = pd.concat(ret_map, axis=1).dropna(how="any")
    corr = ret_df.corr()
    G = nx.Graph()
    G.add_nodes_from(ret_map.keys())
    for a in corr.index:
        for b in corr.columns:
            if a != b and abs(corr.at[a, b]) >= threshold:
                G.add_edge(a, b, weight=float(corr.at[a, b]))
    return G

def construct_graph(feats: dict):
    # Static sector graph could go here if desired
    G = dynamic_correlation_graph(
        feats, 
        window=config.DYNAMIC_CORR_WINDOW, 
        threshold=config.CORRELATION_THRESHOLD
    )
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    path = os.path.join(config.GRAPHS_DIR, "corr_edgelist.csv")
    nx.to_pandas_edgelist(G).to_csv(path, index=False)
    print(f"💾 Dynamic correlation edgelist → {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G
