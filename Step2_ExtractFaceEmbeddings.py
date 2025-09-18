import os
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# Initialize FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define paths
cropped_faces_path = "faces/"  # Path to cropped faces
classes = os.listdir(cropped_faces_path)

# Prepare embeddings and labels
embeddings = []
labels = []
label_map = {cls: idx for idx, cls in enumerate(classes)}  # Map class names to integers
print("Label Mapping:", label_map)

# Process each class folder
for cls in classes:
    class_path = os.path.join(cropped_faces_path, cls)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

            # Extract face embedding
            with torch.no_grad():
                embedding = resnet(img_tensor).squeeze(0).cpu().numpy()

            embeddings.append(embedding)
            labels.append(label_map[cls])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert to numpy arrays for training
embeddings = np.array(embeddings)
labels = np.array(labels)

print("Embeddings Shape:", embeddings.shape)  # (Number of images, 512)
print("Labels Shape:", labels.shape)          # (Number of images,)

# Save embeddings and labels to disk
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
print("Embeddings and labels saved to disk as 'embeddings.npy' and 'labels.npy'.")
