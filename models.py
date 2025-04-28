import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import config

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        layers    = config.NUM_LAYERS
        self.convs.append(GCNConv(in_dim,    hid_dim))
        for _ in range(layers-2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
        self.convs.append(GCNConv(hid_dim, out_dim))
        for _ in range(layers-1):
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
            self.drops.append(torch.nn.Dropout(0.3))

    def forward(self, x, edge_index):
        for conv, bn, drop in zip(self.convs[:-1], self.bns, self.drops):
            x = F.relu(drop(bn(conv(x, edge_index))))
        return self.convs[-1](x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        layers    = config.NUM_LAYERS
        self.convs.append(SAGEConv(in_dim,    hid_dim))
        for _ in range(layers-2):
            self.convs.append(SAGEConv(hid_dim, hid_dim))
        self.convs.append(SAGEConv(hid_dim, out_dim))
        for _ in range(layers-1):
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
            self.drops.append(torch.nn.Dropout(0.3))

    def forward(self, x, edge_index):
        for conv, bn, drop in zip(self.convs[:-1], self.bns, self.drops):
            x = F.relu(drop(bn(conv(x, edge_index))))
        return self.convs[-1](x, edge_index)

class GATWithAtt(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2, heads=config.GAT_HEADS):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, dropout=0.2)
        self.bn1  = torch.nn.BatchNorm1d(hid_dim*heads)
        self.drop = torch.nn.Dropout(0.3)
        self.gat2 = GATConv(hid_dim*heads, out_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x1, (ei, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(self.drop(self.bn1(x1)))
        x2, (ei, att2) = self.gat2(x1, edge_index, return_attention_weights=True)
        return x2, att2

class TemporalGAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=config.HID_DIM, out_dim=2, heads=config.GAT_HEADS):
        super().__init__()
        W = config.FEATURE_WINDOW
        self.window    = W
        self.num_feats = in_dim // W
        self.tconv     = torch.nn.Conv1d(self.num_feats, 16, kernel_size=3, padding=1)
        self.bn0       = torch.nn.BatchNorm1d(16*W)
        self.drop0     = torch.nn.Dropout(0.3)
        self.gat1      = GATConv(16*W, hid_dim, heads=heads, dropout=0.2)
        self.bn1       = torch.nn.BatchNorm1d(hid_dim*heads)
        self.drop1     = torch.nn.Dropout(0.3)
        self.gat2      = GATConv(hid_dim*heads, out_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        N,_ = x.size()
        x   = x.view(N, self.num_feats, self.window)
        x   = F.relu(self.tconv(x))
        x   = x.view(N, -1)
        x   = self.drop0(self.bn0(x))
        h, (ei, att1) = self.gat1(x, edge_index, return_attention_weights=True)
        h  = F.elu(self.drop1(self.bn1(h)))
        out, (ei, att2) = self.gat2(h, edge_index, return_attention_weights=True)
        return out, att2

def get_model(name, input_dim):
    return {
        "GCN":         lambda: GCN(input_dim),
        "GraphSAGE":   lambda: GraphSAGE(input_dim),
        "GATWithAtt":  lambda: GATWithAtt(input_dim),
        "TemporalGAT": lambda: TemporalGAT(input_dim),
    }[name]()