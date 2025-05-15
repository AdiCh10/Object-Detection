import cv2
import random
from ultralytics import YOLO

# Load the YOLOv8 model (replace with your own model path if needed)
model = YOLO("yolov8n.pt")

# Open the COCO class list
with open("utils/coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Generate random colors for each class
detection_colors = []
for _ in class_list:
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    detection_colors.append((b, g, r))

# Initialize the camera (0 is typically the default camera on most laptops)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the camera")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame (optional for performance)
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection on the frame
    detection_output = model(frame, conf=0.45, save=False)

    # Process the detection results
    for result in detection_output:
        boxes = result.boxes  # List of boxes
        
        for box in boxes:
            # Extract box details
            bb = box.xyxy.numpy()[0]  # Bounding box coordinates
            clsID = int(box.cls.numpy()[0])  # Class ID
            conf = box.conf.numpy()[0]  # Confidence score

            # Draw bounding box on the frame
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )

            # Display class name and confidence score
            label = f"{class_list[clsID]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    # Display the frame with detection boxes
    cv2.imshow("Real-Time Object Detection", frame)

    # Break gracefully
    if cv2.waitKey(1) & 0xFF == ord('0'):  # Press '0' to exit early
        print("Exiting...")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
