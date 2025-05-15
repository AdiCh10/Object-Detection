from ultralytics import YOLO
import numpy

# Load a pretrained Yolov8n model
model = YOLO("yolov8n.pt","v8")

# Predict on an image
detection_output = model.predict(source="inference/images/img0.JPG", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())