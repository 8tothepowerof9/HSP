import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class RGCModel(nn.Module):
    def __init__(
        self, in_channels, hidden_dims, out_channels, dropout_rate=0.5, name="rgc"
    ):
        super().__init__()

        self.name = name
        self.layers = nn.ModuleList()
        current_dim = in_channels

        for hidden_dim in hidden_dims:
            self.layers.append(GCNConv(current_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Batch normalization in the second group
        if len(hidden_dims) > 1:
            self.layers.append(nn.BatchNorm1d(hidden_dims[1]))

        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], out_channels)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for _, layer in enumerate(self.layers):
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
                x = F.leaky_relu(x, negative_slope=0.2)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
