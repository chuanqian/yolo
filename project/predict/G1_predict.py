from ultralytics import YOLO

# Load a model
# model = YOLO('../yolov8/yolo11n.pt')  # load an official model
model = YOLO('runs/detect/train2/weights/best.pt')  # load a custom model

# Predict with the model
results = model(source="predict/G1_org/", line_width=3, save=True, workers=2, batch=4, imgsz=1280)  # predict on an image