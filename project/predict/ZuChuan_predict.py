import os
import cv2
from ultralytics import YOLO

# Load a model
# model = YOLO('../yolov8/yolo11n.pt')  # load an official model
model = YOLO('G1_G2_all_1127add.pt')  # load a custom model

# Predict with the model
# results = model(source="/mnt/sdb/ZQC/datasets/ZuChuanProject/1129Test", save=True, workers=2, batch=4, imgsz=1280)  # predict on an image
# Predict with the model
# results = model("/mnt/sdb/ZQC/datasets/ZaDaiDatasetAll/1127_G1_org/", imgsz=1280, conf=0.5, show_conf=False,line_width=3, show_boxes=False, save=False)  # predict on an image
images_path = r"/mnt/sdb/ZQC/datasets/ZuChuanProject/1129Test"
save_path = r"/mnt/sdb/ZQC/datasets/ZuChuanProject/1129Results"
os.makedirs(save_path, exist_ok=True)
image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img in image_files:
    img_dir = os.path.join(images_path, img)
    source = cv2.imread(img_dir)
    results = model(source=source, imgsz=1280, conf=0.5, show_conf=False, line_width=1, show_boxes=False, save=False)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    # save_name = str(uuid.uuid4())
    cv2.imwrite(os.path.join(save_path, img), annotated_frame)
