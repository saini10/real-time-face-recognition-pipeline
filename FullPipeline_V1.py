import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from ultralytics import YOLO  # YOLOv8 for person and cup detection

# Define the face classifier model
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

# Load YOLOv8 model for detecting persons and cups
yolo_model = YOLO("yolov8n.pt")  # Use pretrained YOLOv8 model

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MTCNN (for face detection) and InceptionResnetV1 (for embedding extraction)
mtcnn = MTCNN(keep_all=False, device=device)  # Face detector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # FaceNet

# Load trained face classifier
classifier = FaceClassifier(input_dim=512, num_classes=3).to(device)
classifier.load_state_dict(torch.load("face_classifier50Epochs.pth", map_location=device))
classifier.eval()

# Define the label map
label_map = {0: "prabhu", 1: "shubh", 2: "unknown"}

# Define confidence threshold for "unknown" classification
UNKNOWN_THRESHOLD = 0.8

# Open video capture (webcam or video file)
video = cv2.VideoCapture(0)  # Use 0 for webcam or specify video file path
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Detect persons and cups using YOLOv8
    results = yolo_model(frame, conf=0.5)  # Set confidence threshold for YOLO
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections

    # Process detected objects
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)

        # Check if the detected object is a person or a cup
        if class_id == 0:  # Class 0 is 'person' in COCO dataset
            # Extract the person region
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]

            # Detect face within the person ROI using MTCNN
            img = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            face = mtcnn(img)
            if face is not None:
                # Convert face to tensor and move to device
                face = face.unsqueeze(0).to(device)

                # Extract embedding for the detected face
                with torch.no_grad():
                    embedding = resnet(face).squeeze(0).cpu().numpy()

                # Classify the embedding
                with torch.no_grad():
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device).unsqueeze(0)
                    output = classifier(embedding_tensor)
                    probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
                    max_prob = np.max(probabilities)
                    label_idx = np.argmax(probabilities)

                    # Classify as "unknown" if confidence is below the threshold
                    if max_prob < UNKNOWN_THRESHOLD:
                        label = "unknown"
                    else:
                        label = label_map[label_idx]

                # Display the classification result on the video frame
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        elif class_id == 41:  # Class 41 is 'cup' in COCO dataset
            # Draw bounding box for the cup
            cv2.putText(frame, "cup", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Show the video frame with detections
    cv2.imshow("Real-Time Face Recognition with YOLO", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video.release()
cv2.destroyAllWindows()
