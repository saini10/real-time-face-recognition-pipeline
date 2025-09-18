import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load the trained classifier model
class FaceClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
mtcnn = MTCNN(keep_all=False, device=device)  # Face detector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # FaceNet
classifier = FaceClassifier(input_dim=512, num_classes=3).to(device)
classifier.load_state_dict(torch.load("face_classifier.pth", map_location=device))
classifier.eval()

# Label mapping
label_map = {0: "prabhu", 1: "shubh", 2: "unknown"}

# Video capture
video = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    # Detect face
    face = mtcnn(img)
    if face is not None:
        # Convert face to tensor and move to device
        face = face.unsqueeze(0).to(device)

        # Extract embedding
        with torch.no_grad():
            embedding = resnet(face).squeeze(0).cpu().numpy()

        # Classify face
        with torch.no_grad():
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device).unsqueeze(0)
            output = classifier(embedding_tensor)
            label_idx = torch.argmax(output, axis=1).item()
            label = label_map[label_idx]

        # Display result on the frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
