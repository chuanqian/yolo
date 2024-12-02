from ultralytics import YOLO

# Load a model
# model = YOLO("../yolov8/yolov8l-seg.pt")  # load an official model
model = YOLO("runs/train2/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
