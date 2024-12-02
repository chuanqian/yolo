from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO('../model_config/box_model.yaml').load('/mnt/sdb/ZQC/pre_training/yolov8n.pt')  # build from YAML and transfer weights

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="../data_config/box_detect.yaml", epochs=200, workers=2, batch=4, imgsz=1280, project="/mnt/sdb/ZQC/yolo_weights/detect/box_detect")
