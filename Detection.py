import cv2
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import torch

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8n (Nano) for fast performance

# Initialize MTCNN for face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Open video capture (replace 'test_video.mp4' with 0 for webcam)
cap = cv2.VideoCapture(0)  # Replace with '0' for webcam

# Colors for labels
colors = {
    "person": (0, 255, 0),  # Green
    "cup": (255, 0, 0),     # Blue
    "face": (0, 0, 255)     # Red
}

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO for object detection
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cls = int(cls)  # Class index

        # Label for the detected class
        label = None
        if cls == 0:  # Person
            label = "Person"
        elif cls == 41:  # Example: Class 41 for "Cup" (YOLO's class list)
            label = "Cup"

        if label:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[label.lower()], 2)
            # Add label text
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label.lower()], 2)

            # If it's a person, process further for face detection
            if label == "Person":
                # Crop the detected person region
                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # Detect face using MTCNN
                faces, _ = mtcnn.detect(person_crop)
                if faces is not None:
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face
                        # Adjust face bounding box to the original frame size
                        fx1, fy1, fx2, fy2 = int(fx1 + x1), int(fy1 + y1), int(fx2 + x1), int(fy2 + y1)

                        # Draw bounding box around the face
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), colors["face"], 2)
                        # Add face label
                        cv2.putText(frame, "Face", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["face"], 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
