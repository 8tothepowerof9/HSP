from .graph_dataset import GraphHandSignDataset
from .std_dataset import StdHandSignDataset
from .utils import *
from .config import *
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader


DATASET_LIST = {"graph": GraphHandSignDataset, "std": StdHandSignDataset}
DATALOADER_TYPE = {"graph": GeoDataLoader, "std": DataLoader}
