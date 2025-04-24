# graph_construction.py

import os
import numpy as np
import pandas as pd
import networkx as nx
import config

CORR_THRESHOLD = 0.3

def construct_graph(feats: dict):
    """
    Build a static correlation graph over all tickers based on
    their 1d returns, thresholded at CORR_THRESHOLD.
    Save as edge list CSV.
    """
    dates = None
    # gather returns_1d into DataFrame
    ret_df = pd.concat(
        {t: f["returns_1d"] for t, f in feats.items()},
        axis=1
    ).dropna(how="any")
    if ret_df.empty:
        print("❌ No features found! Make sure `prepare_features()` ran correctly.")
        return None

    corr = ret_df.corr()
    G = nx.Graph()
    for t in feats.keys():
        G.add_node(t)
    # add edges above threshold
    for i, a in enumerate(corr.index):
        for b in corr.columns[i+1:]:
            if corr.at[a, b] >= CORR_THRESHOLD:
                G.add_edge(a, b, weight=float(corr.at[a, b]))
    # save edgelist
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    out_path = os.path.join(config.GRAPHS_DIR, "correlation_edgelist.csv")
    nx.to_pandas_edgelist(G).to_csv(out_path, index=False)
    print(f"💾 Graph edgelist → {out_path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G
