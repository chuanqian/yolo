from ultralytics import YOLO
import os
import cv2
# Load a model
# model = YOLO('../yolov8/yolo11n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model(source="", save=True, workers=2, batch=4, imgsz=1280)  # predict on an image

