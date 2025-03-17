import os
import sys
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from dataset import *
from models import *
from trainers import GraphTrainer
from evals import GNNEvaluator

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


evaluator = GNNEvaluator(model, None, device, save=False)

image_path = "tests/B.jpg"
predicted_sign = evaluator.static_predict(image_path)
print(f"Predicted Sign: {predicted_sign}")

evaluator.realtime_predict()
