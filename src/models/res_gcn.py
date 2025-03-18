import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .base import BaseModel


class RGCModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.name = config["model"]["name"]
        self.lr = config["train"]["lr"]
        self.weight_decay = config["train"]["weight_decay"]
        self.betas = tuple(config["train"]["betas"])

        in_channels = 4
        hidden_dims = config["model"]["hidden_dims"]
        out_channels = 2  # TODO: Replace with number of classes later
        dropout_rate = config["model"]["dropout_rate"]

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

    def get_criterion(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

    def get_optimizer(self):
        return torch.nn.CrossEntropyLoss()
