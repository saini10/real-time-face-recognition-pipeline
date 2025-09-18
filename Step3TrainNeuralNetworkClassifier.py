import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load embeddings and labels
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Dynamically set the device (GPU/TPU if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move data to device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Define the neural network
class FaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Model parameters
input_dim = embeddings.shape[1]  # Size of embeddings (512)
num_classes = len(np.unique(labels))  # Number of classes (prabhu, shubh, unknown)
model = FaceClassifier(input_dim, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store loss for each epoch
loss_values = []

# Training the model
epochs = 30
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store the loss
    loss_values.append(loss.item())

    # Print loss for each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test.cpu(), y_pred_classes.cpu())
    print(f"Test Accuracy: {accuracy:.4f}")

# Plot the loss values
plt.plot(range(1, epochs + 1), loss_values, marker='o')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "face_classifier.pth")
print("Model saved as face_classifier.pth")
