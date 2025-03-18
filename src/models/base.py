from torch import nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def get_criterion(self):
        raise NotImplementedError
