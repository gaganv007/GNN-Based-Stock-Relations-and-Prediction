# models.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GATWithAtt(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hid_dim*heads, out_dim, heads=1, concat=False, dropout=0.1)
    def forward(self, x, edge_index):
        # first layer with attention weights
        x1, (edge_idx1, attn1) = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        # second layer
        x2, (edge_idx2, attn2) = self.gat2(x1, edge_index, return_attention_weights=True)
        # return logits + final attention scores
        return x2, attn2

def get_model(name, input_dim):
    return {
        "GCN":         lambda: GCN(input_dim),
        "GraphSAGE":   lambda: GraphSAGE(input_dim),
        "GATWithAtt":  lambda: GATWithAtt(input_dim),
    }[name]()
