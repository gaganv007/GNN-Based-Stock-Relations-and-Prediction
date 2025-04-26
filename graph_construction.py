import os
import pandas as pd
import networkx as nx
import config
from datetime import timedelta

def dynamic_correlation_graph(feats, window, threshold):
    common_dates = set.intersection(*(set(df.index) for df in feats.values()))
    if not common_dates:
        return nx.Graph()

    end_date = max(common_dates)
    start_date = end_date - timedelta(days=window)
    ret_dict = {}
    for t, df in feats.items():
        col = next((c for c in df.columns if c.startswith("returns_1d_t-0")), None)
        if col is None:
            continue
        series = df[col].loc[(df.index >= start_date) & (df.index <= end_date)]
        if not series.empty:
            ret_dict[t] = series

    if not ret_dict:
        return nx.Graph()

    ret_df = pd.DataFrame(ret_dict).dropna(how="any")
    corr   = ret_df.corr()

    G = nx.Graph()
    G.add_nodes_from(ret_df.columns)
    for i, a in enumerate(corr.index):
        for j, b in enumerate(corr.columns):
            if i < j and abs(corr.iat[i, j]) >= threshold:
                G.add_edge(a, b, weight=float(corr.iat[i, j]))
    return G

def construct_graph(feats):
    Gs = dynamic_correlation_graph(feats, config.SHORT_CORR_WINDOW, config.CORRELATION_THRESHOLD)
    Gl = dynamic_correlation_graph(feats, config.LONG_CORR_WINDOW,  config.CORRELATION_THRESHOLD)

    G = nx.Graph()
    G.add_nodes_from(feats.keys())
    for u, v, d in Gs.edges(data=True):
        G.add_edge(u, v, weight_short=d["weight"])
    for u, v, d in Gl.edges(data=True):
        if G.has_edge(u, v):
            G[u][v]["weight_long"] = d["weight"]
        else:
            G.add_edge(u, v, weight_long=d["weight"])

    for sector, ticks in config.SECTOR_MAP.items():
        G.add_node(sector)
        for t in ticks:
            if t in feats:
                G.add_edge(sector, t, weight_hyper=1.0)

    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    pd.DataFrame(
        [{"source": u, "target": v, **attrs}
         for u, v, attrs in G.edges(data=True)]
    ).to_csv(os.path.join(config.GRAPHS_DIR, "full_edgelist.csv"), index=False)

    return G
