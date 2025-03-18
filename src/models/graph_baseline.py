import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .base import BaseModel


class GraphBaseline(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.lr = config["train"]["lr"]
        self.weight_decay = config["train"]["weight_decay"]
        self.betas = tuple(config["train"]["betas"])

        self.name = config["model"]["name"]
        in_channels = 4  # [x, y, z, angle]
        hidden_dim = config["model"]["hidden_dim"]
        out_channels = 2  # TODO: Replace with number of classes later

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

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
