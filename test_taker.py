from skimage.measure import label
import cv2
import os, copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tr
from segmentation.model import UNet, WPU_Net
from segmentation.tools import posprecess, stitch, proprecess_img, tailor, download_from_url
import time
from segmentation.evaluation import eval_RI_VI, eval_F_mapKaggle
from segmentation.COLOR_SHOW_CONFRONTATION import color_confrontation 
import shutil
import datetime
def inference(cwd,dataset_dir):
    '''
    predict results and evaluate on test samples for segmentation with wpu-net
    '''
    # transform = tr.Compose([
    #     tr.ToTensor(),
    #     tr.Normalize(mean=[0.9336267],  # RGB
    #                  std=[0.1365774])
    # ])
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path=os.path.join(cwd, "DPU_model_"+str(current_time))

    model_path = os.path.join(cwd,checkpoint_path, '_best_model_state.pth')
    # cwd = os.getcwd()#这句话获取了
    output_test_dir = os.path.join(dataset_dir,"test-dev")
    # generate overlapping cropped image for test
    output_test_crop_dir = os.path.join(output_test_dir, "test_overlap_crop")
    [os.makedirs(os.path.join(output_test_crop_dir, dir_name), exist_ok=True) for dir_name in ['', 'JPEGImages', 'Annotations']]
    result_output_dir=os.path.join(dataset_dir, "results_DPU"+str(current_time), "label_results")
    # if not os.path.exists(cwd):
    #     try:
    #         download_from_url(url='https://doc-0k-3k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/c4ag025396gf90cp0rt7vcaci3ml4teo/1567137600000/03563112468744709654/*/1Gc2j-DrJhX0E4fnvRItf95o0BXWQa-wr?e=download',
    #                       filename='wpu_net_parameters.zip', save_dir=os.path.join(cwd, 'segmentation'))
    #     except:
    #         raise Exception('download failed, please check the url')
            
    
   
    # result_save_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "test_overlap_crop",'labels')
    # result_total_save_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "result")
    # [os.makedirs(dir, exist_ok=True) for dir in [result_save_dir, result_total_save_dir]]

    model = UNet(num_channels=2, num_classes=2, multi_layer=True)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    # 先加载模型参数dict文件
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    # 加载修改后的新参数dict文件
    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load(model_path))
    model.eval()
    for video in sorted(os.listdir(os.path.join(output_test_dir, 'JPEGImages'))):
        [os.makedirs(os.path.join(output_test_crop_dir, i, video), exist_ok=True) for i in ['JPEGImages', 'Annotations']]
        print('you are processing video ', video)
        # video的名字是video1,video2,video3...
        # ///////////////////////////////////////////////////
        # video的名字是video1,video2,video3...
        for item in sorted(os.listdir(os.path.join(output_test_dir, 'Annotations', video))):
            # 新建一个label文件夹用于反复存储测试结果而不改变原文件夹
            src = os.path.join(output_test_dir, 'Annotations', video, item)
            dst = os.path.join(result_output_dir,'Annotations', video, item)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # 将原本的test label的第一张图复制到这个新文件夹
            shutil.copy(src, dst)
        # 完成复制。
        # ////////////////////////////////////////////////////
        # print(sorted(os.listdir(os.path.join(result_output_dir, 'labels', video))))
        for item in sorted(os.listdir(os.path.join(output_test_dir, 'JPEGImages', video))):
            tailor(256, 256, os.path.join(output_test_dir, 'JPEGImages', video, item), os.path.join(output_test_crop_dir, 'JPEGImages', video), region=32)
        for item in sorted(os.listdir(os.path.join(result_output_dir, 'JPEGImages', video))):
            tailor(256, 256, os.path.join(result_output_dir, 'JPEGImages', video, item), os.path.join(output_test_crop_dir, 'JPEGImages', video), region=32)

        # inferece cropped images
        images = sorted(os.listdir(os.path.join(output_test_crop_dir, "JPEGImages", video)))
        ori_images = sorted(os.listdir(os.path.join(output_test_dir, "JPEGImages", video)))
        print(len(ori_images))
        start_time = time.time()
        count = 0
        min_file = int(ori_images[0].split(".")[0])
        max_file = int(ori_images[-1].split(".")[0])
        print((ori_images))
        # min_file = 149  # 1  117  149
        # max_file = 296  # 116  148  296
        downsample_folder=os.path.join(cwd,"temporary_data","downsample_tensors",str(video))
        os.makedirs(downsample_folder,exist_ok=True)
        for directions in [1,-1]:#'forward','backward'
            if directions==1:
                count = 0
                for item in images:
                    if item.endswith(".png"):
                        filename = item.split(".")[0]
                        pic_num = item.split("_")[0]
                        if int(pic_num) > min_file and int(pic_num) < max_file:
                            count += directions
                            test_image = os.path.join(output_test_crop_dir, "JPEGImages", video, filename + ".png")
                            img = proprecess_img(test_image)
                            # last mask
                            last_name = str(int(pic_num) - directions).zfill(3)+'_'+filename.split('_')[1] + '_' + filename.split('_')[2]
                            last_mask = cv2.imread(os.path.join(output_test_crop_dir, "Annotations", video, last_name + ".png"), 0)
                            last_tensor = torch.Tensor(np.array(last_mask)).unsqueeze(0).unsqueeze(0)
                            last_tensor[last_tensor == 255] = -6
                            last_tensor[last_tensor == 0] = 1
                            
                            output,downsample_result=model(img, last_tensor)
                            # torch.save(downsample_result,os.path.join(downsample_folder,"img_no."+str(filename)+"_of this batch_downsample_result.pt"))
                            # output = model(inputs=img, last=last_tensor, opposite_result=downsample_result)
                            result_npy = posprecess(output, close=True)
                            cv2.imwrite(os.path.join(output_test_crop_dir, "Annotations" , video ,filename + ".png"), result_npy)

            # elif directions==-1:
                # count = max_file               
                # for item in reversed(images):
                #     if item.endswith(".png"):
                #         filename = item.split(".")[0]
                #         pic_num = item.split("_")[0]
                #         if int(pic_num) > min_file and int(pic_num) < max_file:
                #             count += directions
                #             test_image = os.path.join(output_test_crop_dir, "images", video, filename + ".png")
                #             img = proprecess_img(test_image)
                #             # last mask
                #             last_name = str(int(pic_num) - directions).zfill(3)+'_'+filename.split('_')[1] + '_' + filename.split('_')[2]
                #             last_mask = cv2.imread(os.path.join(output_test_crop_dir, "labels", video, last_name + ".png"), 0)
                #             last_tensor = torch.Tensor(np.array(last_mask)).unsqueeze(0).unsqueeze(0)
                #             last_tensor[last_tensor == 255] = -6
                #             last_tensor[last_tensor == 0] = 1
                #             downsample_result=torch.load(os.path.join(downsample_folder,"img_no."+str(filename)+"_of this batch_downsample_result.pt"))
                #             output= model(img, last_tensor, downsample_result)
                #             result_npy = posprecess(output, close=True)
                #             cv2.imwrite(os.path.join(output_test_crop_dir, "labels" , video ,filename + ".png"), result_npy)
            
                            
        end_time = time.time()
        average_time = (end_time - start_time) / count
        print("end ...", average_time)

        # stitch cropped images
        imgList = sorted(os.listdir(os.path.join(output_test_dir, "JPEGImages", video)))
        print(len(imgList))
        n = 0
        for img in imgList:
            if img.endswith(".png"):
                name = img.split(".")[0]
                if int(name) > min_file and int(name) <= max_file:
                    print('you are stitching picture ', name)
                    stitch(256, 256, name, os.path.join(output_test_crop_dir,'Annotations', video), os.path.join(result_output_dir, 'Annotations', video, name + ".png"), 32)
                    n += 1
    # color_confrontation(os.path.join(cwd, "datasets","segmentation"))
    print("end...")

    # evaluate
    # RI_save_dir = os.path.join(cwd, 'segmentation', 'evaluations', 'big_RI_VI')
    # Map_save_dir = os.path.join(cwd, 'segmentation', 'evaluations', 'big_F_mAP')

    # [os.makedirs(dir,exist_ok=True) for dir in [RI_save_dir, Map_save_dir]]
    # print(checkpoint_path + " model " + "#####" * 20)



    # #可选功能：测量RI+VI, F+mAP
    # #（这两个函数非常耗时，不想测则不建议使用）
    # eval_RI_VI(os.path.join(cwd, checkpoint_path),  os.path.join(RI_save_dir, checkpoint_path + ".txt"), gt_dir=os.path.join(cwd, checkpoint_path))
    # eval_F_mapKaggle(os.path.join(cwd, checkpoint_path), os.path.join(Map_save_dir, checkpoint_path + ".txt"), gt_dir=os.path.join(cwd, checkpoint_path))




