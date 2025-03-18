import torch
import torch.nn as nn
import torch.optim as optim

class VectorExtractor(nn.Module):
    """ Extracts features from hand vectors (orientation, movement, etc.) """
    def __init__(self, input_dim, hidden_dim=32):
        super(VectorExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class LandmarkAngleExtractor(nn.Module):
    """ Extracts features from hand landmarks and calculated joint angles """
    def __init__(self, input_dim, hidden_dim=32):
        super(LandmarkAngleExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class FusionModel(nn.Module):
    """ Combines extracted features from both branches and makes a prediction """
    def __init__(self, vector_dim, landmark_angle_dim, num_classes=10):
        super(FusionModel, self).__init__()
        self.vector_extractor = VectorExtractor(vector_dim)
        self.landmark_angle_extractor = LandmarkAngleExtractor(landmark_angle_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32, 64),  # Merge both feature extractors
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Final classification layer
            nn.Softmax(dim=1)  # Probability output
        )

    def forward(self, vector_input, landmark_angle_input):
        v_feat = self.vector_extractor(vector_input)
        l_feat = self.landmark_angle_extractor(landmark_angle_input)
        fused = torch.cat((v_feat, l_feat), dim=1)
        return self.fusion(fused)



# Define dataset properties
num_samples = 100
vector_dim = 6  # Example: Hand orientation vectors
landmark_angle_dim = 84  # Example: 21 landmarks (x, y, z) + 21 angles

# Generate random training data
vector_data = torch.rand(num_samples, vector_dim)
landmark_angle_data = torch.rand(num_samples, landmark_angle_dim)
labels = torch.randint(0, 10, (num_samples,))  # 10-class classification

# Convert labels to one-hot encoding
labels = nn.functional.one_hot(labels, num_classes=10).float()

# Initialize model, loss function, and optimizer
model = FusionModel(vector_dim, landmark_angle_dim, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(vector_data, landmark_angle_data)

    # Compute loss
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training completed!")
