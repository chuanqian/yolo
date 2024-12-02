import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Load a model
model = YOLO('runs/train2/train/weights/best.pt')  # load a custom trained

# Predict with the model
# results = model("/mnt/sdb/ZQC/datasets/ZaDaiDatasetAll/1127_G1_org/", imgsz=1280, conf=0.5, show_conf=False,line_width=3, show_boxes=False, save=False)  # predict on an image
images_path = r"/mnt/sdb/ZQC/datasets/ZaDaiDatasetAll/1127_G1_org/"
save_path = r"predict/1127_1"
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

# output_dir = "ZaDaiResults/"
# os.makedirs(output_dir, exist_ok=True, imgsz=1280, conf=0.5, show_conf=False,line_width=3, show_boxes=False, save=False)
# # # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     path = result.path
#     path = output_dir + path.split("/")[-1]
#     result.save(filename=path)  # save to disk