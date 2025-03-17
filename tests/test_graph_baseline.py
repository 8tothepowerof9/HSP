import os
import sys
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import numpy as np
import networkx as nx
from torch_geometric.data import Data

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from dataset import *
from models import *
from trainers import GraphTrainer

train_dataset = GraphHandSignDataset(split="train", ds_dir="data")
test_dataset = GraphHandSignDataset(split="test", ds_dir="data")

# Create DataLoader for batch training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
in_channels = 4  # [x, y, z, angle]
hidden_dim = 64
out_channels = 2  # Number of hand sign classes

# Initialize model
# model = GraphBaseline(in_channels, hidden_dim, out_channels).to(device)
model = RGCModel(in_channels, [64, 64, 64], out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

trainer = GraphTrainer(model, criterion, optimizer, 40, scheduler, device, save=False)

history = trainer.fit(train_loader, test_loader)


# Load an image, apply mediapipe on it, and visualize the graph
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append(
                        [lm.x, lm.y, lm.z if hasattr(lm, "z") else 0]
                    )  # (x, y, z)
            return np.array(landmarks), image  # Return landmarks and original image

    return None, image  # No hand detected


def image_to_graph(image_path):
    landmarks, image = extract_hand_landmarks(image_path)

    if landmarks is None:
        print("No hand detected in the image.")
        return None, image

    # Step 1: Normalize (translate & scale)
    landmarks = translate_landmarks(landmarks, 0)  # Center the hand at the wrist
    joint_angles = cal_all_finger_angles(landmarks)  # Compute angles
    landmarks = scale_landmarks(landmarks, max_dist=1)  # Scale

    # Step 2: Construct edges using HAND_CONNECTIONS
    G = nx.Graph()
    for idx, landmark in enumerate(landmarks):
        G.add_node(idx, pos=(landmark[0], landmark[1], landmark[2]))

    for edge in mp_hands.HAND_CONNECTIONS:
        G.add_edge(edge[0], edge[1])

    # Convert adjacency matrix to COO format
    adjacency_matrix = nx.adjacency_matrix(G)
    coo = adjacency_matrix.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

    # Step 3: Create node features (x, y, z, angle)
    features = np.hstack([landmarks, np.zeros((landmarks.shape[0], 1))])

    for joint, (p1, p2, p3) in FINGER_JOINTS.items():
        angle = joint_angles.get(joint, 0)
        features[p2, -1] = angle  # Assign angle to middle joint

    x = torch.tensor(features, dtype=torch.float)

    # Create PyG Data object
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data, image


def visualize_hand_landmarks(image_path):
    landmarks, image = extract_hand_landmarks(image_path)

    if landmarks is None:
        print("No hand detected in the image.")
        return

    for idx, (x, y, _) in enumerate(landmarks):
        h, w, _ = image.shape
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Draw landmark

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def predict_hand_sign(model, image_path, device):
    model.eval()  # Set model to evaluation mode
    graph_data, image = image_to_graph(image_path)

    if graph_data is None:
        return "No hand detected"

    graph_data = graph_data.to(device)

    with torch.no_grad():
        output = model(graph_data)
        pred = output.argmax(dim=1).item()  # Get predicted class index

    predicted_label = [key for key, val in LABEL_MAPPING.items() if val == pred][0]
    return predicted_label


image_path = "tests/B.jpg"
visualize_hand_landmarks(image_path)

predicted_sign = predict_hand_sign(model, image_path, device)
print(f"Predicted Sign: {predicted_sign}")
