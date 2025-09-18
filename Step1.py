import os
from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance
import torch
import numpy as np

# Initialize MTCNN for face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)  # keep_all=False for single largest face

# Define dataset paths
input_dataset_path = "dataset/"  # Path to original dataset
output_dataset_path = "faces/"  # Path to save cropped faces
os.makedirs(output_dataset_path, exist_ok=True)

# Get class folders (e.g., 'shubh', 'prabhu', 'unknown')
classes = os.listdir(input_dataset_path)

# Process each class folder
for cls in classes:
    input_class_path = os.path.join(input_dataset_path, cls)
    output_class_path = os.path.join(output_dataset_path, cls)
    os.makedirs(output_class_path, exist_ok=True)  # Create class folder in output

    for img_name in os.listdir(input_class_path):
        img_path = os.path.join(input_class_path, img_name)
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Adjust brightness to make features more visible
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.5)  # Increase brightness by 1.5x

            # Detect face and crop
            face = mtcnn(img)

            if face is not None:
                # Convert face tensor to NumPy array with proper scaling
                face = face.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
                face = np.clip(face * 255, 0, 255).astype(np.uint8)  # Scale to 0–255 and clip values

                # Save cropped face
                output_path = os.path.join(output_class_path, img_name)
                Image.fromarray(face).save(output_path)
                print(f"Saved cropped face to {output_path}")
            else:
                print(f"No face detected in {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
