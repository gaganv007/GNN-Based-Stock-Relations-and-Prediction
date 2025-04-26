import os
import pandas as pd
import networkx as nx
import config
from datetime import timedelta

def dynamic_correlation_graph(feats, window, threshold):
    common = set.intersection(*(set(df.index) for df in feats.values()))
    end_date = max(common)
    start_date = end_date - timedelta(days=window)
    ret_map = {
        t: feats[t]['returns_1d'].loc[start_date:end_date]
        for t in feats
        if 'returns_1d' in feats[t]
    }
    ret_df = pd.concat(ret_map, axis=1).dropna(how="any")
    corr = ret_df.corr()
    G = nx.Graph()
    G.add_nodes_from(ret_map.keys())
    for i, a in enumerate(corr.index):
        for j, b in enumerate(corr.columns):
            if i < j and abs(corr.iat[i, j]) >= threshold:
                G.add_edge(a, b, weight=corr.iat[i, j])
    return G

def construct_graph(feats):
    Gs = dynamic_correlation_graph(feats, config.SHORT_CORR_WINDOW, config.CORRELATION_THRESHOLD)
    Gl = dynamic_correlation_graph(feats, config.LONG_CORR_WINDOW, config.CORRELATION_THRESHOLD)
    G = nx.Graph()
    G.add_nodes_from(feats.keys())
    for u, v, data in Gs.edges(data=True):
        G.add_edge(u, v, weight_short=data['weight'])
    for u, v, data in Gl.edges(data=True):
        if G.has_edge(u, v):
            G[u][v]['weight_long'] = data['weight']
        else:
            G.add_edge(u, v, weight_long=data['weight'])
    for sector, ticks in config.SECTOR_MAP.items():
        G.add_node(sector)
        for t in ticks:
            if t in feats:
                G.add_edge(sector, t, weight_hyper=1.0)
    os.makedirs(config.GRAPHS_DIR, exist_ok=True)
    nx.to_pandas_edgelist(G).to_csv(os.path.join(config.GRAPHS_DIR, "full_edgelist.csv"), index=False)
    return G