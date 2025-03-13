import os
import sys
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from dataset import GraphHandSignDataset


def visualize_graph(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        G.add_edge(src, dst)

    pos = {
        i: data.x[i][:2].numpy() for i in range(data.x.shape[0])
    }  # Use (x, y) as position

    plt.figure(figsize=(5, 5))
    nx.draw(
        G, pos, with_labels=True, node_size=300, node_color="skyblue", edge_color="gray"
    )
    plt.show()


ds = GraphHandSignDataset(split="train")
print(f"Dataset size: {len(ds)}")

# Get random sample from 0 to 100
idx = np.random.randint(0, 100)
sample = ds[idx]
print(sample)

print("Node features (x):")
print(sample.x)  # Should print a tensor of shape (21, 4)

print("Edge index (connectivity):")
print(sample.edge_index)  # Should print edges in COO format

print("Label: ")
print(sample.y)  # Should print a tensor with the label

loader = DataLoader(ds, batch_size=32, shuffle=True)
for batch in loader:
    print(batch)
    break  # Print the first batch and stop

visualize_graph(sample)
