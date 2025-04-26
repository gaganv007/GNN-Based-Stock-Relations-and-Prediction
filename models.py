import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import config

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.bn1   = torch.nn.BatchNorm1d(hid_dim)
        self.drop1 = torch.nn.Dropout(0.3)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.drop1(self.bn1(self.conv1(x, edge_index))))
        return self.conv2(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.bn1   = torch.nn.BatchNorm1d(hid_dim)
        self.drop1 = torch.nn.Dropout(0.3)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.drop1(self.bn1(self.conv1(x, edge_index))))
        return self.conv2(x, edge_index)

class GATWithAtt(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2, heads=4):
        super().__init__()
        self.gat1  = GATConv(in_dim, hid_dim, heads=heads, dropout=0.1)
        self.bn1   = torch.nn.BatchNorm1d(hid_dim * heads)
        self.drop1 = torch.nn.Dropout(0.3)
        self.gat2  = GATConv(hid_dim * heads, out_dim, heads=1, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        x1, (ei1, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(self.drop1(self.bn1(x1)))
        x2, (ei2, att2) = self.gat2(x1, edge_index, return_attention_weights=True)
        return x2, att2

class TemporalGAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2, heads=4):
        super().__init__()
        W = config.FEATURE_WINDOW
        if in_dim % W != 0:
            W = 1
        self.window = W
        self.num_feats = in_dim // W

        self.tconv = torch.nn.Conv1d(
            in_channels=self.num_feats,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.bn0   = torch.nn.BatchNorm1d(16 * self.window)
        self.drop0 = torch.nn.Dropout(0.3)

        self.gat1  = GATConv(16 * self.window, hid_dim, heads=heads, dropout=0.1)
        self.bn1   = torch.nn.BatchNorm1d(hid_dim * heads)
        self.drop1 = torch.nn.Dropout(0.3)
        self.gat2  = GATConv(hid_dim * heads, out_dim, heads=1, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        N, _ = x.size()
        x = x.view(N, self.num_feats, self.window)
        x = F.relu(self.tconv(x))
        x = x.view(N, -1)
        x = self.drop0(self.bn0(x))

        h, (ei1, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        h = F.elu(self.drop1(self.bn1(h)))
        out, (ei2, att2) = self.gat2(h, edge_index, return_attention_weights=True)
        return out, att2

def get_model(name, input_dim):
    return {
        "GCN":         lambda: GCN(input_dim),
        "GraphSAGE":   lambda: GraphSAGE(input_dim),
        "GATWithAtt":  lambda: GATWithAtt(input_dim),
        "TemporalGAT": lambda: TemporalGAT(input_dim),
    }[name]()
