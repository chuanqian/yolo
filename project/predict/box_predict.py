# from ultralytics import YOLO
#
# # Load a model
# # model = YOLO('../yolov8/yolo11n.pt')  # load an official model
# model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model
#
# # Predict with the model
# results = model(source="test/board_box_labeling_two.mp4", save_crop=True, save_frames=True, save=True, workers=2, batch=4, imgsz=1280)  # predict on an image
import uuid

import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Open the video file
video_path = "test/board_box_labeling_two.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
image_save_path = r"predict/images-labels"
# labels_save_path = r"predict/labels/"
os.makedirs(image_save_path, exist_ok=True)
# os.makedirs(labels_save_path, exist_ok=True)
save_name = 1
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, conf=0.4)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # save_name = str(uuid.uuid4())
        results[0].save_txt(os.path.join(image_save_path, str(save_name) + ".txt"))
        cv2.imwrite(os.path.join(image_save_path, str(save_name) + ".jpg"), annotated_frame)
        save_name += 1
        # Display the annotated frame
        # cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

path = image_save_path
# path = r'F:\work\2023-0103-0106\part\exp\labels'         # jpg图片和对应的生成结果的txt标注文件，放在一起
# path3 = r'F:\work\2023-0103-0106\part\exp\labels\cut'    # 裁剪出来的小图保存的根目录
path3 = "predict/cut"
os.makedirs(path3, exist_ok=True)
# path2 = r'F:\work\2023-0103-0106\part\exp\labels\crop'   # 覆盖目标区域后的原图
 
file = os.listdir(path)
# 生成图像与标签名称列表
img_total = []
txt_total = []
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":                      # 图片的后缀名
        img_total.append(first)
    else:
        txt_total.append(first)

# 排序
img_total = sorted(img_total, key=lambda x: int(x.split('.')[0]))
txt_total = sorted(txt_total, key=lambda x: int(x.split('.')[0]))

n = 1
for img_name in img_total:
    if img_name in txt_total:
        filename_img = img_name+".jpg"
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)
        h, w = img.shape[0], img.shape[1]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  # resize 图像大小，否则roi区域可能会报错
        filename_txt = img_name+".txt"
        with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                coordinate = line.split(" ")
                x_center = w * float(coordinate[1])       # coordinate[1]左上点的x坐标
                y_center = h * float(coordinate[2])       # coordinate[2]左上点的y坐标
                width = int(w*float(coordinate[3]))       # coordinate[3]图片width
                height = int(h*float(coordinate[4]))      # coordinate[4]图片height
                lefttopx = int(x_center-width/2.0)
                lefttopy = int(y_center-height/2.0)
                filename_last = str(n) + ".jpg"
                roi = img[lefttopy+1:lefttopy+height+3, lefttopx+1:lefttopx+width+1]
                cv2.imwrite(os.path.join(path3, filename_last), roi)
                # filename_last = img_name+"_"+str(n)+".jpg"    # 裁剪出来的小图文件名
                # img[lefttopy + 1:lefttopy + height + 3, lefttopx + 1:lefttopx + width + 1] = (255, 255, 255)
                n = n+1
            # cv2.imwrite(os.path.join(path2, filename_last), img)
    else:
        continue

cut_image_path = path3

# 获取保存的图片文件路径
cut_image_files = [f for f in os.listdir(cut_image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
cut_image_files = sorted(cut_image_files, key=lambda x: int(x.split('.')[0]))

# 检查是否有图像文件
if not cut_image_files:
    print("文件夹中没有找到图像文件")
    exit()

model_lsdy = YOLO("/mnt/sdb/ZQC/yolo_detect/ultralytics_detect/project/nut_detect/runs/detect/train/weights/best.pt")
cut_n = 1
nut_detect_save_dir = "predict/nut_detect_save"
# os.makedirs(nut_detect_save_dir, exist_ok=True)

# for cut_img in cut_image_files:
#     cut_img_dir = os.path.join(cut_image_path, cut_img)
#     lsdy_results = model_lsdy(cut_img_dir, imgsz=1280, workers=4, batch=8)
#     lsdy_annotated_frame = lsdy_results[0].plot()
#     cv2.imwrite(os.path.join(nut_detect_save_dir, str(cut_n) + ".jpg"), lsdy_annotated_frame)
#     cut_n += 1
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# 获取保存的图片文件路径
# image_files = [f for f in os.listdir(nut_detect_save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
# # print(image_files)
# # exit(0)
# # 检查是否有图像文件
# if not image_files:
#     print("文件夹中没有找到图像文件")
#     exit()

# # print(image_files)

# # 读取第一张图像以获取尺寸（假设所有图像尺寸一致）
# first_image_path = os.path.join(nut_detect_save_dir, image_files[0])
# first_image = cv2.imread(first_image_path)
# height, width, _ = first_image.shape

# output_video_dir = "predict/lsd_lsdy_Generate1280.mp4"

# # 创建视频写入对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式
# video_writer = cv2.VideoWriter(output_video_dir, fourcc, 25, (width, height))

# # 遍历所有图像文件并将其添加到视频中
# for image_file in image_files:
#     image_path = os.path.join(nut_detect_save_dir, image_file)
#     image = cv2.imread(image_path)

#     if image is not None:
#         video_writer.write(image)  # 将图像写入视频
#     else:
#         print(f"无法读取图像: {image_path}")

# # 释放 VideoWriter 对象
# video_writer.release()

# print(f"视频已成功保存到: {output_video_dir}")

output_video_dir = "predict/lsd_lsdy_Generate3.mp4"

# 获取保存的图片文件路径
image_files = [f for f in os.listdir(nut_detect_save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
# print(image_files)
# exit(0)
# 检查是否有图像文件
if not image_files:
    print("文件夹中没有找到图像文件")
    exit()

# 读取第一张图像以获取尺寸（假设所有图像尺寸一致）
first_image_path = os.path.join(nut_detect_save_dir, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

img_size = (480, 640)


# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式
video_writer = cv2.VideoWriter(output_video_dir, fourcc, fps, img_size)

# 遍历所有图像文件并将其添加到视频中
for image_file in image_files:
    image_path = os.path.join(nut_detect_save_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)

    if image is not None:
        video_writer.write(image)  # 将图像写入视频
    else:
        print(f"无法读取图像: {image_path}")

# 释放 VideoWriter 对象
video_writer.release()

print(f"视频已成功保存到: {nut_detect_save_dir}")







# # 读取第一张图像以获取尺寸（假设所有图像尺寸一致）
# first_image_path = os.path.join(cut_image_path, cut_image_files[0])
# first_image = cv2.imread(first_image_path)
# height, width, _ = first_image.shape


# # 创建视频写入对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式
# video_writer = cv2.VideoWriter(cut_image_path, fourcc, fps, (width, height))







# cut_file = os.listdir(cut_image_path)
# cut_images = []
# for filename in cut_file:
#     first, last = os.path.splitext(filename)
#     if last == ".jpg":                      # 图片的后缀名
#         cut_images.append(first)

# cut_images = sorted(cut_images, key=lambda x: int(x.split('.')[0]))
# model_lsdy = YOLO("/mnt/sdb/ZQC/yolo_detect/ultralytics_detect/project/nut_detect/runs/detect/train/weights/best.pt")
# for cut_img in cut_images:
#     filename_img = cut_img+".jpg"
#     cut_img_dir = os.path.join(path, filename_img)
#     cut_img_numpy = cv2.imread(path1)
#     lsdy_results = model_lsdy(cut_img_numpy, conf=0.4, imgsz=1280)
    



