import os
from PIL import Image

def val_crop_create(ori_folder,save_folder):
# 定义文件夹A和文件夹B的路径
    for folder in (['images','labels']):
        for SUBfolder in os.listdir(os.path.join(ori_folder,folder)):
            folder_A = os.path.join(ori_folder,folder,SUBfolder)
            folder_B = os.path.join(save_folder,folder,SUBfolder)
            # '/root/data/videoanno/WPU-Net-master/datasets/segmentation/net_train/val_crop/images/video2'

            # 创建文件夹B（如果不存在）
            os.makedirs(folder_B, exist_ok=True)

            # 遍历文件夹A中的所有图片文件
            for filename in os.listdir(folder_A):
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif') :
                    # 打开图片文件
                    image_path = os.path.join(folder_A, filename)
                    image = Image.open(image_path)

                    # 获取图片的宽度和高度
                    width, height = image.size

                    # 计算每个切割后的图像的宽度和高度
                    sub_width = width // 4
                    sub_height = height // 4

                    # 切割图片并保存到文件夹B中
                    for i in range(4):
                        for j in range(4):
                            # 计算切割后图像的区域
                            left = j * sub_width
                            upper = i * sub_height
                            right = (j + 1) * sub_width
                            lower = (i + 1) * sub_height

                            # 切割图像
                            sub_image = image.crop((left, upper, right, lower))

                            # 构建切割后图像的文件名
                            sub_filename = f"{filename[:-4].zfill(3)}_{j}_{i}.png"

                            # 保存切割后的图像到文件夹B中
                            sub_image.save(os.path.join(folder_B, sub_filename))

                    # 关闭图片文件
                    image.close