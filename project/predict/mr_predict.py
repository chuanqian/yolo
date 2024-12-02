import os

from ultralytics import YOLO

# Load a model
model = YOLO('../runs/segment/train2/weights/best.pt')  # load a custom trained

# Predict with the model
results = model("", imgsz=1280, conf=0.5, show_conf=False, show_boxes=False, save=False)  # predict on an image
output_dir = "MRResults/"
os.makedirs(output_dir, exist_ok=True)
# # Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    path = result.path
    path = output_dir + path.split("/")[-1]
    result.save(filename=path)  # save to disk
