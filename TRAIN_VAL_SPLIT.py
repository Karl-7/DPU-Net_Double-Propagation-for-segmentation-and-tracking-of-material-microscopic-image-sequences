import os
import random
import shutil

# 设置文件夹路径

def train_val_split(train_val_folder):
    # 创建train和val文件夹
    os.makedirs(os.path.join(train_val_folder, 'train', 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(train_val_folder, 'train', 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(train_val_folder, 'val', 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(train_val_folder, 'val', 'JPEGImages'), exist_ok=True)
    annotations_dir=os.path.join(train_val_folder, 'Annotations')
    jpegimages_dir=os.path.join(train_val_folder, 'JPEGImages')
    train_dir=os.path.join(train_val_folder, 'train')
    val_dir=os.path.join(train_val_folder, 'val')
    # 获取所有视频文件名
    video_files = os.listdir(annotations_dir)
    # 随机打乱视频文件顺序
    random.shuffle(video_files)

    # 计算分割点
    split_point = int(len(video_files) * 0.9)

    # 将前split_point个视频移动到train文件夹
    for video_name in video_files[:split_point]:
        annotations_src = os.path.join(annotations_dir, video_name)
        annotations_dst = os.path.join(train_dir, 'Annotations', video_name)
        jpegimages_src = os.path.join(jpegimages_dir, video_name)
        jpegimages_dst = os.path.join(train_dir, 'JPEGImages', video_name)
        shutil.move(annotations_src, annotations_dst)
        shutil.move(jpegimages_src, jpegimages_dst)

    # 将剩余的视频移动到val文件夹
    for video_name in video_files[split_point:]:
        annotations_src = os.path.join(annotations_dir, video_name)
        annotations_dst = os.path.join(val_dir, 'Annotations', video_name)
        jpegimages_src = os.path.join(jpegimages_dir, video_name)
        jpegimages_dst = os.path.join(val_dir, 'JPEGImages', video_name)
        shutil.move(annotations_src, annotations_dst)
        shutil.move(jpegimages_src, jpegimages_dst)