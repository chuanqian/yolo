from ultralytics import YOLO

# 
model = YOLO('/mnt/sdb/ZQC/pre_training/yolov8s-cls.pt')  # 加载预训练模型（推荐用于训练）

# 使用2个GPU训练模型
# in
results = model.train(data='/mnt/sdb/ZQC/project/ZCClassify/ZCClassifyDataset', epochs=300, imgsz=224, device=0, batch=8,
                      project='/mnt/sdb/ZQC/yolo_weights/classify/ZCClassify_classify')
