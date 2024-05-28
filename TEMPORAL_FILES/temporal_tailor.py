from segmentation.tools import tailor
import os
# this file is to tail one video only. it should only be used for creating the val_crop set
path='/root/data/videoanno/WPU-Net-master/datasets/segmentation/net_train/val'
store_path='/root/data/videoanno/WPU-Net-master/datasets/segmentation/net_train/val_crop'
image_folders=os.path.join(path, 'images')
label_folders=os.path.join(path, 'labels')
image_list = os.listdir(image_folders)  # Fix: Use os.listdir() directly on image_folders
label_list = os.listdir(label_folders)  # Fix: Use os.listdir() directly on label_folders
for i in range(len(image_list)):  # Fix: Use range(len(image_list)) to iterate over indices
    image_path = os.path.join(image_folders, image_list[i])
    label_path = os.path.join(label_folders, label_list[i])
    for item in sorted(os.listdir(image_path)):
        tailor(256,256,os.path.join(image_path,item), os.path.join(store_path,"images", image_list[i]))
        tailor(256,256,os.path.join(label_path,item), os.path.join(store_path,"labels", image_list[i]))