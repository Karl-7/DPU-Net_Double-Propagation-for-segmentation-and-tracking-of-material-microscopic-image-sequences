import os
from PIL import Image

def train_create(ori_folder):
    # 修改ori_folder的文件夹名称
    save_folder= ori_folder
    save_folder = ori_folder.rstrip(os.path.sep) + '_crop'
    
    # 定义文件夹A和文件夹B的路径
    for folder in (['JPEGImages', 'Annotations']):
        for SUBfolder in os.listdir(os.path.join(ori_folder, folder)):
            folder_A = os.path.join(ori_folder, folder, SUBfolder)
            folder_B = os.path.join(save_folder, folder, SUBfolder)
            # 创建文件夹B（如果不存在）
            os.makedirs(folder_B, exist_ok=True)
            # 遍历文件夹A中的所有图片文件
            image_files = [filename for filename in os.listdir(folder_A) if filename.endswith(('.png', '.jpg', '.tif'))]
            num_images = len(image_files)
            num_folders = (num_images // 30) + 1
            # 获取新文件夹的起始编号
            # existing_folders = os.listdir(folder_A)
            # existing_videos = [folder for folder in existing_folders if folder.startswith('video')]
            # if existing_videos:
            #     last_video = max(existing_videos, key=lambda x: int(x[5:]))
            #     start_video_num = int(last_video[5:]) + 1
            # else:
                # start_video_num = 1
            # 分割图片并保存到新文件夹中
            for i in range(num_folders):
                # 创建新文件夹
                # new_folder_name = f"video{start_video_num + i}"
                new_folder_name = f"video{i+1}"
                new_folder_path = os.path.join(save_folder, folder, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                # 计算每个新文件夹中的图片数量
                start_index = i * 30
                end_index = min((i + 1) * 30, num_images)
                num_images_in_folder = end_index - start_index
                # 切割并重命名图片
                for j in range(num_images_in_folder):
                    # 获取原始图片路径
                    image_filename = image_files[start_index + j]
                    image_path = os.path.join(folder_A, image_filename)
                    # 打开图片文件
                    image = Image.open(image_path)
                    # 获取图片的宽度和高度
                    width, height = image.size
                    # 计算每个切割后的图像的宽度和高度
                    sub_width = width // 2
                    sub_height = height // 2
                    # 切割图片并保存到新文件夹中
                    for k in range(2):
                        for l in range(2):
                            # 计算切割后图像的区域
                            left = l * sub_width
                            upper = k * sub_height
                            right = (l + 1) * sub_width
                            lower = (k + 1) * sub_height
                            # 切割图像
                            sub_image = image.crop((left, upper, right, lower))
                            # 构建切割后图像的文件名
                            sub_filename = f"{str(j).zfill(3)}_{l}_{k}.png"
                            # 保存切割后的图像到新文件夹中
                            sub_image.save(os.path.join(new_folder_path, sub_filename))
                    # 关闭图片文件
                    image.close()
