from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO('../model_config/G1_model.yaml').load('/mnt/sdb/ZQC/pre_training/yolov8m.pt')  # build from YAML and transfer weights

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="../data_config/G1.yaml", epochs=600, workers=4, batch=8, imgsz=1280, project="/mnt/sdb/ZQC/yolo_weights/detect/G1_detect")
