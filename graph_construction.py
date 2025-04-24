# graph_construction.py
import numpy as np
import networkx as nx
import config

def build_static_graph(price_dict):
    """
    price_dict: ticker -> price-series (aligned numpy.ndarray)
    returns: networkx Graph with nodes=stocks, weighted edges by corr>THRESH
    """
    tickers = list(price_dict.keys())
    prices  = np.stack([price_dict[t] for t in tickers], axis=1)
    corr    = np.corrcoef(prices, rowvar=False)
    G = nx.Graph()
    for i, a in enumerate(tickers):
        G.add_node(i, ticker=a)
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            if corr[i,j] >= config.CORR_THRESHOLD:
                G.add_edge(i,j, weight=float(corr[i,j]))
    return G
