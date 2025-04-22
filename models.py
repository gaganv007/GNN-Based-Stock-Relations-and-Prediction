import torch
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc    = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc    = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc    = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class TemporalGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.lstm    = LSTM(input_dim, hidden_dim, batch_first=True)
        self.gcn     = GCNConv(hidden_dim, hidden_dim)
        self.fc      = Linear(hidden_dim, output_dim)

    def forward(self, batch_list):
        xs = []
        for data in batch_list:
            x = data.x.unsqueeze(0)             # [1, nodes, features]
            out, _ = self.lstm(x)               # [1, seq_len, hidden_dim]
            xs.append(out[:, -1, :])            # take last output
        x = torch.cat(xs, 0)                   # [batch, hidden_dim]
        edge_index = batch_list[-1].edge_index
        x = F.relu(self.gcn(x, edge_index))
        return self.fc(x)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=5):
        super().__init__()
        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def get_model(name, input_dim, hidden_dim, output_dim):
    return {
        'GCN':      GCNModel,
        'GAT':      GATModel,
        'GraphSAGE':GraphSAGEModel,
        'TemporalGNN':TemporalGNNModel,
        'LSTM':     LSTMModel
    }[name](input_dim, hidden_dim, output_dim)
