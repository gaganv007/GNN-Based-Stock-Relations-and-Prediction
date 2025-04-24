import os
import pandas as pd
import networkx as nx
import config

def construct_graph(feats: dict, corr_thr: float = config.CORRELATION_THRESHOLD):
    """
    Build static correlation graph based on each ticker’s 1-day returns.
    Saves edge list CSV under config.GRAPHS_DIR.
    """
    # collect every ticker’s returns_1d series
    series_map = {}
    for t, df in feats.items():
        if "returns_1d" in df.columns:
            series_map[t] = df["returns_1d"]
    if not series_map:
        print("❌ No 'returns_1d' series found, skipping graph construction.")
        return None

    ret_df = pd.concat(series_map, axis=1).dropna(how="any")
    corr   = ret_df.corr()

    G = nx.Graph()
    G.add_nodes_from(series_map.keys())

    # add edges above threshold
    for a in corr.index:
        for b in corr.columns:
            if a != b and corr.at[a, b] >= corr_thr:
                G.add_edge(a, b, weight=float(corr.at[a, b]))

    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    out_path = os.path.join(config.GRAPHS_DIR, "correlation_edgelist.csv")
    nx.to_pandas_edgelist(G).to_csv(out_path, index=False)

    print(f"💾 Graph edgelist → {out_path}  "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G
