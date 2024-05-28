import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TEST_METRIC.TEST_METRIC import test_metric_logger
from segmentation.val_crop_create import val_crop_create
from segmentation.train_create import train_create
# from TRAIN_PARAS import train_parameters
from TRAIN_VAL_SPLIT import train_val_split
import datetime
class train_test():
    def __init__(self,dataset_address,model):
        super().__init__()
        self.data_path=dataset_address
        self.cwd =os.path.dirname(os.path.abspath(__file__))#'/root/data/videoanno/WPU-Net-master/'
        self.two_way_propagation=False
        self.num_workers=1
        self.trainval_set=os.path.join(self.data_path,"trainval")#train set address
        self.test_set=os.path.join(self.data_path,"test-dev")

        # self.models_are_at=
        self.trainer=os.path.join(self.cwd,"trainer.py")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.test_save_folder = "DPU_RESULTS"+str(current_time)
        self.folder_name="DPU_RESULTS"+str(current_time)
        self.logger_name="DPU_LOGGER"+str(current_time)+".txt"


        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if model == "DPU-Net":
            self.trainer=os.path.join(self.cwd,"trainer_DPU.py")
            self.test_save_folder = "DPU_RESULTS"+str(current_time)
            self.folder_name="DPU_RESULTS"+str(current_time)
            self.logger_name="DPU_LOGGER"+str(current_time)+".txt"
        elif model == "WPU-Net":
            self.trainer=os.path.join(self.cwd,"trainer_WPU.py")
            self.test_save_folder = "WPU_RESULTS"+str(current_time)
            self.folder_name="WPU_RESULTS"+str(current_time)
            self.logger_name="WPU_LOGGER"+str(current_time)+".txt"
        elif model=="U-Net":
            self.trainer=os.path.join(self.cwd,"trainer_U.py")
            self.test_save_folder = "U_RESULTS"+str(current_time)
            self.folder_name="U_RESULTS"+str(current_time)
            self.logger_name="U_LOGGER"+str(current_time)+".txt"
        
    def run_train(self):
        
        if not os.path.exists(os.path.join(self.trainval_set,"train")) or not os.path.exists(os.path.join(self.trainval_set,"val")):
            train_val_split(train_val_folder=self.trainval_set)
        if not os.path.exists(os.path.join(self.trainval_set,"train_crop")):
        # val_crop_create(ori_folder=os.path.join(self.train_set,"val"),save_folder=os.path.join(self.train_set,"val_crop"))
            train_create(ori_folder=os.path.join(self.trainval_set,"train"))
        if not os.path.exists(os.path.join(self.trainval_set,"val_crop")):
            val_crop_create(ori_folder=os.path.join(self.trainval_set,"val"),save_folder=os.path.join(self.trainval_set,"val_crop"))
        
        torch.cuda.empty_cache()
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     self.test_save_folder = "DPU_RESULTS"+str(current_time)
        #     self.folder_name="DPU_RESULTS"+str(current_time)
        #     self.logger_name="DPU_LOGGER"+str(current_time)+".txt"
        
        os.system(f"""python {self.trainer} --input="{self.trainval_set}" --bs=1 --loss="abw" --epochs 100 --ml --workers=1""")
        
    def run_test(self,model):
        if model == "DPU-Net":
            from test_taker_DPU import inference
        elif model == "WPU-Net":
            from test_taker_WPU import inference
        elif model=="U-Net":
            from test_taker_UNet import inference
        inference(self.cwd,two_way_propagation=self.two_way_propagation, middle_pic_nums=296,save_folder=self.test_save_folder)

    def confrontation(self):

        test_metric_logger(label_path=os.path.join(self.test_set,"test/original_labels/video1"), 
                           test_folder=os.path.join(self.test_set,'test/images/video1'), 
                           result_path=os.path.join(self.data_path,self.folder_name,"label_results/video1"),
                           save_path=os.path.join(self.data_path,self.folder_name,'METRIC_PICS/video1'),
                           logger_path=os.path.join(self.data_path,self.folder_name,self.logger_name))
       