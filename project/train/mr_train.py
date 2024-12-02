from ultralytics import YOLO

# Load a model
# model = YOLO("../MRDatasets/yolov8-seg.yaml")  # build a new model from scratch
# model = YOLO('../yolov8/yolov8l-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('../model_config/mr_model_segment.yaml').load('/mnt/sdb/ZQC/pre_training/yolov8n-seg.pt')  # build from YAML and transfer weights

# Use the model
model.train(data="../data_config/mr_segment.yaml", task="segment", mode="train", imgsz=1280, workers=0, batch=4, epochs=300,
            device=0, project="/mnt/sdb/ZQC/yolo_weights/segment/MagneticRing_segment")  # train the model
