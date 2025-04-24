import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads)
        self.conv2 = GATConv(hid_dim*heads, out_dim, heads=1)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class TemporalGNN(GCN):
    pass

def get_model(name, input_dim):
    return {
        "GCN":        lambda: GCN(input_dim),
        "GAT":        lambda: GAT(input_dim),
        "GraphSAGE":  lambda: GraphSAGE(input_dim),
        "TemporalGNN":lambda: TemporalGNN(input_dim),
    }[name]()
