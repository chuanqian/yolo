from ultralytics import YOLO

# Load a model
# model = YOLO("../MRDatasets/yolov8-seg.yaml")  # build a new model from scratch
# model = YOLO('../yolov8/yolov8l-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('../model_config/screw_gun_model_segment.yaml').load('/mnt/sdb/ZQC/pre_training/yolov8s-seg.pt')  # build from YAML and transfer weights

# Use the model
model.train(data="../data_config/screw_gun_segment.yaml", task="segment", mode="train", imgsz=640, workers=4, batch=8, epochs=300,
            device=0, project="/mnt/sdb/ZQC/yolo_weights/segment/ScrewGun_segment")  # train the model
