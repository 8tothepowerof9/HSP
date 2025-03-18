import os
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
import networkx as nx
from torch_geometric.data import Data, Dataset as GeoDataset
from sklearn.model_selection import train_test_split
from .utils import *
from .config import *


class GraphHandSignDataset(GeoDataset):
    def __init__(
        self, split="train", ds_dir="data", transform=None, pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.ds_dir = ds_dir
        self.split = split
        self.data = []
        self.labels = []

        self.__readdata__()

    def __readdata__(self):
        # Loop through each file in the ds_dir
        for filename in os.listdir(self.ds_dir):
            if filename.endswith(".csv"):
                # Read the file
                df = pd.read_csv(os.path.join(self.ds_dir, filename))  # (200, 63)

                # Validate data
                if df.shape[1] != NUM_LANDMARKS * 3:
                    continue

                label = filename.split(".")[0]

                if label not in LABEL_MAPPING:
                    continue

                # Convert to numpy array and reshape to (200, 21, 3)
                landmarks = df.to_numpy().reshape(-1, NUM_LANDMARKS, 3)

                # Convert to idx
                label = LABEL_MAPPING[label]

                # Append the data and label to the lists
                self.data.extend(landmarks)
                # Append to labels the number of times equal to the number of frames
                self.labels.extend([label] * len(landmarks))

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # Depending on the split, split the data into train and test
        if self.split == "train":
            self.data, _, self.labels, _ = train_test_split(
                self.data, self.labels, test_size=0.25, random_state=RANDOM_STATE
            )
        elif self.split == "test":
            _, self.data, _, self.labels = train_test_split(
                self.data, self.labels, test_size=0.25, random_state=RANDOM_STATE
            )
        else:
            raise ValueError("Invalid split. Choose either 'train' or 'test'.")

    def __construct_graph__(self, landmarks):
        G = nx.Graph()

        for idx, landmark in enumerate(landmarks):
            G.add_node(idx, pos=(landmark[0], landmark[1], landmark[2]))

        connections = mp.solutions.hands.HAND_CONNECTIONS
        for edge in connections:
            G.add_edge(edge[0], edge[1])

        adjacency_matrix = nx.adjacency_matrix(G)

        coo = adjacency_matrix.tocoo()
        edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

        return G, edge_index

    def __len__(self):
        return len(self.data)

    def len(self):
        return len(self.data)

    def get(self, idx):
        frame = self.data[idx]
        label = self.labels[idx]

        # Normalize
        landmarks = translate_landmarks(frame, 0)

        # Compute angles
        joint_angles = cal_all_finger_angles(landmarks)

        landmarks = scale_landmarks(landmarks, max_dist=1)

        # Construct graph
        _, edge_index = self.__construct_graph__(landmarks)

        # Add angle as additional feature
        features = np.hstack([landmarks, np.zeros((landmarks.shape[0], 1))])

        for joint, (_, p2, _) in FINGER_JOINTS.items():
            angle = joint_angles.get(joint, 0)  # Default to 0 if not found
            features[p2, -1] = angle  # Assign angle to middle joint

        # Convert to tensors
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)

        return Data(x=x, y=y, edge_index=edge_index)
