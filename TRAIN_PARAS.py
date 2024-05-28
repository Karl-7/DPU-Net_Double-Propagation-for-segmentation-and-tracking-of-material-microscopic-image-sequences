import os
import glob
import datetime
class train_parameters():
    def __init__(self):
        
        self.cwd =os.path.dirname(os.path.abspath(__file__))#'/root/data/videoanno/WPU-Net-master/'
        # self.dataset=os.path.join(os.path.dirname(self.cwd),'datasets')#dataset address
        
        self.two_way_propagation=False
        self.num_workers=1
        self.trainval_set=os.path.join(self.dataset,"trainval")#train set address
        self.test_set=os.path.join(self.dataset,"test-dev")

        # self.models_are_at=
        self.trainer=os.path.join(self.cwd,"trainer.py")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.test_save_folder = "DPU_RESULTS"+str(current_time)
        self.folder_name="DPU_RESULTS"+str(current_time)
        self.logger_name="DPU_LOGGER"+str(current_time)+".txt"