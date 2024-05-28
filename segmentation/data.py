# DataSet loader class for WPU-Net
# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch.utils.data.dataset import Dataset
import os, time
import torch
import numpy as np
from segmentation.tools import getImg, rand_crop, rand_rotation, rand_horizontalFlip, rand_verticalFlip, dilate_mask, get_expansion
from segmentation.weight_map_loss import caculate_weight_map

class IronDataset(Dataset):

    def __init__(self,video_list,label_list, train=True, transform=None, crop = True, crop_size=(400, 400), dilate = 5,crop_train=False):
        """
        DataSet for our data.
        :param dataset_folder: Address for train set and test set
        :param train:  True if you load train set, False if you load test set
        :param transform:  The pytorch transform you used
        :param crop: Used when you want to randomly crop images
        :param crop_size: Used when randomly cropping images
        :param dilate: Used when dilated boundary is used
        """

        self.__file = []
        self.__im = []
        self.__mask = []
        self.__last = []
        self.__next=[]
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.train = train
        self.dilate = dilate
        self.crop_train=crop_train
        # self.count = forward_or_backward_last_count
        self.video_list = video_list
        
        self.label_list = label_list
        print("video_list",self.video_list)
        print("label_list",self.label_list)
        
        self.current_video_index = 0
        self.load_video(self.video_list[0], self.label_list[0])
        self.frames_per_video=[]
        for video_folder in self.video_list:
            self.frames_per_video.append(len(os.listdir(video_folder)))
        
        # if self.train:
        #     folder = dataset_folder + "/train/"
        # else:
        #     folder = dataset_folder+ "/val_crop/"
        # folder = dataset_folder
        # org_folder = folder + "images/"    # folder for original images
        # mask_folder = folder + "labels/"    # folder for labels

        # Find the largest and smallest image id, training and testing needs to start from the second picture in WPU-Net
    def load_video(self, video_folder, label_folder):
        max_file = 0
        min_file = 10000000
        for file in sorted(os.listdir(video_folder)):
            # print("file",file)
            if file.endswith(".png") or file.endswith(".tif") or file.endswith(".jpg"):
                if self.train:
                    if self.crop_train==True:
                        pic_num = int(os.path.splitext(file)[0].split("_")[0])
                    else:
                        pic_num = int(os.path.splitext(file)[0])
                else:
                    pic_num = int(os.path.splitext(file)[0].split("_")[0])
                if pic_num > max_file:
                    max_file = pic_num
                if pic_num < min_file:
                    min_file = pic_num

        for file in sorted(os.listdir(video_folder)):
            if file.endswith(".png") or file.endswith(".tif") or file.endswith(".jpg"):
                filename = os.path.splitext(file)[0]
                # print("files:",sorted(os.listdir(video_folder)))
                if self.train:
                    if self.crop_train==True:
                        pic_num = int(filename.split("_")[0])
                    else:
                        pic_num = int(filename)
                else:
                    pic_num = int(filename.split("_")[0])
                
                if pic_num != min_file and pic_num!=max_file:
                    # 1. read file name
                    self.__file.append(filename)
                    # 2. read original image
                    self.__im.append(os.path.join(video_folder,file))
                    # 3. read  mask image
                    self.__mask.append(label_folder + "/" + filename+".png")
                    
                    # 4. load last label mask or last result mask
                    if self.train:
                        if self.crop_train==True:
                            file_last = str(int(pic_num) - 1).zfill(3) +  '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                            file_next = str(int(pic_num) + 1).zfill(3) +  '_' + filename.split('_')[1] + '_' + filename.split('_')[2] 
                        else:
                            file_last = str(int(filename) - 1).zfill(3)
                            file_next = str(int(filename) + 1).zfill(3)
                    else:
                        file_last =str(int(pic_num) - 1).zfill(3) +  '_' + filename.split('_')[1] + '_' + filename.split('_')[2] 
                        file_next= str(int(pic_num) + 1).zfill(3) +  '_' + filename.split('_')[1] + '_' + filename.split('_')[2] 
                    self.__last.append(label_folder + "/" + file_last + ".png")
                    self.__next.append(label_folder + "/" + file_next + ".png")
                
        self.dataset_size = len(self.__file)

    def __getitem__(self, index):
        # Calculate the video index and frame index based on the number of frames in each video
        video_index = 0
        while index >= self.frames_per_video[video_index]:
            index -= self.frames_per_video[video_index]
            video_index += 1
        frame_index = index
    # If the requested video is not the current video, load the new video
        if video_index != self.current_video_index:
            self.load_video(self.video_list[video_index], self.label_list[video_index])
            self.current_video_index = video_index
    # Get the frame and the corresponding mask
        img = getImg( self.__im[frame_index])  # Original
        mask = getImg(self.__mask[frame_index])  # mask
        last = getImg(self.__last[frame_index])  # last information
        next = getImg(self.__next[frame_index])  

        # img = getImg(self.__im[index])  # Original
        # mask = getImg(self.__mask[index])  # mask
        # last = getImg(self.__last[index])  # last information
        # print(self.__last[index])
        # 裁剪图像 保证img和mask随机裁剪在同一位置   Crop image, Ensuring that img and mask are randomly cropped in the same location
        if self.train and self.crop:  # , weight 
            img, mask, last, next = rand_crop(data=img, label=mask, last=last, next=next, height=self.crop_size[1], width=self.crop_size[0])
            img, mask, last, next = rand_rotation(data=img, label=mask, last=last, next=next)
            img, mask, last, next = rand_verticalFlip(data=img, label=mask, last=last, next=next)
            img, mask, last, next = rand_horizontalFlip(data=img, label=mask, last=last, next=next)
        
        weight,_ = caculate_weight_map(np.array(mask))
        mask = dilate_mask(np.array(mask), iteration=int((self.dilate-1)/2))
        #kernel=np.ones(self.dilate,self.dilate)
        #mask=dilate_mask(np.array(mask),kernel)
        #last=dilate_mask(np.array(last),kernel)
        if self.transform is not None:
            img = self.transform(img)
            mask = torch.Tensor(np.array(mask)).unsqueeze(0)/255
            last = torch.Tensor(np.array(last)).unsqueeze(0)
            next = torch.Tensor(np.array(next)).unsqueeze(0)
            weight = np.ascontiguousarray(weight, dtype=np.float32)
            weight = torch.from_numpy(weight.transpose((2, 0, 1)))

        return img, mask, last, next, weight

    def __len__(self):
        return len(self.__im)