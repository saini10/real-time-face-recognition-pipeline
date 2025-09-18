import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create directory to save graphs
os.makedirs("Graphs_30_2", exist_ok=True)

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

# Store loss and accuracy for each epoch
loss_values = []
accuracy_values = []

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

    # Evaluate on test set
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = torch.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test.cpu(), y_pred_classes.cpu())
        accuracy_values.append(accuracy)

    # Print loss and accuracy for each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Final evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    final_accuracy = accuracy_score(y_test.cpu(), y_pred_classes.cpu())
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

# Plot and save loss curve
plt.figure()
plt.plot(range(1, epochs + 1), loss_values, marker='o', label="Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("Graphs_30_2/loss_curve.png")
print("Saved Loss Curve: Graphs_30_2/loss_curve.png")

# Plot and save accuracy curve
plt.figure()
plt.plot(range(1, epochs + 1), accuracy_values, marker='o', label="Accuracy", color='green')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.savefig("Graphs_30_2/accuracy_curve.png")
print("Saved Accuracy Curve: Graphs_30_2/accuracy_curve.png")

# Confusion Matrix
cm = confusion_matrix(y_test.cpu(), y_pred_classes.cpu())
class_labels = [f"Class {i}" for i in range(num_classes)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.savefig("Graphs_30_2/confusion_matrix.png")
print("Saved Confusion Matrix: Graphs_30_2/confusion_matrix.png")

# Save the trained model
torch.save(model.state_dict(), "face_classifier_2_Visual_30Epochs_2.pth")
print("Model saved as face_classifier.pth")
