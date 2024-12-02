# import os
#
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('../runs/segment/train/weights/best.pt')  # load a custom trained
#
# # Predict with the model
# results = model("", batch=4, imgsz=1280, conf=0.5, show_conf=False, show_boxes=False, save=False)  # predict on an image
# output_dir = "ScrewGunResults/"
# os.makedirs(output_dir, exist_ok=True)
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

import os

# 视频流处理
import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("runs/segment/train/weights/best.pt")

# Open the video file
video_path = "predict/LSDPredict.mp4"
output_video_dir = "predict/LSDGenerate640.mp4"
output_image_dir = "predict/output_images640"
os.makedirs(output_image_dir, exist_ok=True)
# output_video_path = 'predict/LSDGenerate.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高

n = 1

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, conf=0.5)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        # cv2.imshow("YOLO Inference", annotated_frame)
        cv2.imwrite(os.path.join(output_image_dir, str(n) + ".jpg"), annotated_frame)
        n += 1
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# 获取保存的图片文件路径
image_files = [f for f in os.listdir(output_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
# print(image_files)
# exit(0)
# 检查是否有图像文件
if not image_files:
    print("文件夹中没有找到图像文件")
    exit()

# 读取第一张图像以获取尺寸（假设所有图像尺寸一致）
first_image_path = os.path.join(output_image_dir, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape


# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式
video_writer = cv2.VideoWriter(output_video_dir, fourcc, fps, (width, height))

# 遍历所有图像文件并将其添加到视频中
for image_file in image_files:
    image_path = os.path.join(output_image_dir, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        video_writer.write(image)  # 将图像写入视频
    else:
        print(f"无法读取图像: {image_path}")

# 释放 VideoWriter 对象
video_writer.release()

print(f"视频已成功保存到: {output_image_dir}")
