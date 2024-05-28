import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
sys.path.append(os.getcwd())
from segmentation.data_WPU import IronDataset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from segmentation.model import UNet, WPU_Net
from segmentation.weight_map_loss import WeightMapLoss
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from skimage import morphology
import argparse
from TRAIN_SETTINGS import settings
import glob
from torch.utils.data import ConcatDataset
import shutil
# python videoanno/WPU-Net-master/trainer.py --input="/root/data/datasets/segmentation/net_train" --bs=12 --loss="abw" --epochs 500 --ml
#/root/data/videoanno/WPU-Net-master/trainer.py
# cwd = os.getcwd()
cwd=settings().cwd
# print(cwd)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 通过设置随机种子的办法使得pytorch 实验效果可以复制，即使是使用了随机裁剪
# set random seed to make experiments repeatable
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2019)

def plot_imgs(title, imgs):
    plt.figure(figsize=(30, 30))
    plt.subplot(231)
    plt.imshow(imgs[0], cmap='gray')
    plt.axis('off')
    plt.title(title + '_img-label-out-skle')
    plt.subplot(232)
    plt.imshow(imgs[1], cmap='gray')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(imgs[2], cmap='gray')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(imgs[3])
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(imgs[4], cmap='gray')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(imgs[5], cmap='gray')
    plt.axis('off')
    plt.savefig(checkpoint_path + str(title) + '.png')
    plt.close()

def plot(epoch, loss_list, test_loss_list):
    clear_output(True)
    plt.figure()
    plt.title('epoch %s. train loss: %s. val_loss: %s' % (epoch, loss_list[-1], test_loss_list[-1]))
    plt.plot(loss_list, color="r", label="train loss")
    plt.plot(test_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.savefig(checkpoint_path + 'test_model_loss_state.png')
    #plt.show()
    plt.close()

def check(dataset):  # check the output of dataset
    print(dataset.__len__())
    image, mask, last, weight = dataset.__getitem__(2)
    print(image.size(), " ", mask.size(), ' ', np.unique(mask), " ", last.size(), " ", np.unique(last), " ", weight.size(), " ", np.unique(weight))

    test_image = image.squeeze().numpy()
    test_mask = mask.squeeze().numpy()
    last_mask = last.squeeze().numpy()
    test_weight = weight.squeeze().numpy()

    print(np.max(test_image), " ", np.min(test_image))

    print(" last mask ", np.unique(last_mask))
    print(" weight ", np.unique(weight[0, :, :]))

    # plt.figure(figsize=(30, 30))
    # plt.subplot(2, 2, 1), plt.imshow(test_image, cmap="gray")
    # plt.subplot(2, 2, 2), plt.imshow(test_mask, cmap="gray")
    # plt.subplot(2, 2, 3), plt.imshow(last_mask, cmap="gray")
    # plt.subplot(2, 2, 4), plt.imshow(test_weight[1, :, :], vmin=0, vmax=30, cmap="plasma")
    # plt.show()


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 10))
    if not lr < 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# 初始化参数 init model parameters
def weights_init(m):
    classname = m.__class__.__name__  # 得到网络层的名字，如ConvTranspose2d
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight.data)


class Trainer(object):

    def __init__(self, bs=24, loss_type='abw', multi_layer=True, data_dir = '',workers=1):
        # parameteres
        self.batch_size = bs  #24
        self.learning_rate = 1e-4
        self.loss_type = loss_type   #'cbw'
        self.multi_layer = multi_layer
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.val_dir = os.path.join(data_dir, "val_crop")
        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean = [0.9336267],
                        std = [0.1365774])
        ])

        print(' batch_size is ', self.batch_size, ' learning_rate is ', self.learning_rate, ' loss_type is ', self.loss_type, ' multi_layer is ', self.multi_layer, ' data_dir is ', self.data_dir)

        train_vid_list=[os.path.join(self.train_dir,"JPEGImages",vid) for vid in os.listdir(os.path.join(self.train_dir,"JPEGImages"))]
        # train_label_list=(os.listdir(os.path.join(self.train_dir,"Annotations")))
        train_label_list=[os.path.join(self.train_dir,"Annotations",vid) for vid in os.listdir(os.path.join(self.train_dir,"Annotations"))]
        # print("train_vid_list",train_vid_list)
        forward_train_vid=(IronDataset(train_vid_list,train_label_list, train=True, transform=self.transform, crop=True, crop_size=(256, 256), dilate=3,forward_or_backward_last_count=1))
        #backward_train_vid=(IronDataset(train_vid_list,train_label_list, train=True, transform=self.transform, crop=True, crop_size=(256, 256), dilate=3,forward_or_backward_last_count=-1))
        print("train_video_list",forward_train_vid.video_list)
        # check(self.train_dataset)
        self.forward_train_loader = DataLoader(forward_train_vid, batch_size=self.batch_size, shuffle=True, num_workers=workers)
        #self.backward_train_loader = DataLoader(backward_train_vid, batch_size=self.batch_size, shuffle=True, num_workers=workers)
        # load and check test_dataset
        val_vid_list=[os.path.join(self.val_dir,"JPEGImages",vid) for vid in os.listdir(os.path.join(self.val_dir,"JPEGImages"))]
        
        val_label_list=[os.path.join(self.val_dir,"Annotations",vid) for vid in os.listdir(os.path.join(self.val_dir,"Annotations"))]
        forward_val_vid=(IronDataset(val_vid_list,val_label_list, train=False, transform=self.transform, crop=False, crop_size=(256, 256), dilate=3,forward_or_backward_last_count=1))
        #backward_val_vid=(IronDataset(val_vid_list,val_label_list, train=False, transform=self.transform, crop=False, crop_size=(256, 256), dilate=3,forward_or_backward_last_count=-1))
        print("val_video_list",forward_val_vid.video_list)
        #check(self.test_dataset)
        self.forward_test_loader = DataLoader(forward_val_vid, batch_size=self.batch_size, shuffle=False, num_workers=workers)
        #self.backward_test_loader = DataLoader(backward_val_vid, batch_size=self.batch_size, shuffle=False, num_workers=workers)
        print("Training Data : ", len(self.forward_train_loader.dataset))
        print("Test Data :", len(self.forward_test_loader.dataset))

        # set model
        self.model = WPU_Net(num_channels=2, num_classes=2, multi_layer=self.multi_layer)
        print(self.model)
        self.model.apply(weights_init)

        if torch.cuda.is_available():
            model = nn.DataParallel(self.model).cuda()

        if self.loss_type == 'abw':
            self.criterion=WeightMapLoss(_dilate_cof=5)
            self.method = 4
        else:
            self.weight = torch.Tensor([1, 20])
            if torch.cuda.is_available():
                self.weight = self.weight.cuda()
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-5)

    def train(self, epoch, loss_list, plotimgs,temporary_folder):
        self.model.train()
        adjust_learning_rate(self.optimizer, epoch, learning_rate=self.learning_rate)
        downsample_folder=os.path.join(temporary_folder,"train","epoch_"+str(int(epoch)))
        os.makedirs(downsample_folder,exist_ok=True)
    
        F=self.forward_train_loader
        

        iterations = max(1, int(len(F.dataset) / self.batch_size))

        for batch_idx, (image, mask, last, weight) in enumerate(F):
            images = []
            label = mask.squeeze(1).long()

            if torch.cuda.is_available():
                img, label, last, weight = image.cuda(), label.cuda(), last.cuda(), weight.cuda()

            self.optimizer.zero_grad()

            # normalize
            last[last == 255] = -6
            last[last == 0] = 1
            output,downsample_result = self.model(img, last)
            # torch.save(downsample_result,os.path.join(downsample_folder,"batch_no."+str(batch_idx)+"_of this epoch.pt"))


            if self.loss_type == 'abw':
                loss = self.criterion(output, label, weight, method=self.method)
            else:
                loss = self.criterion(output, label)

            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if (batch_idx + 1) % iterations == 0:
                loss_list.append(float(loss.item()))

                # Display the output image of the intermediate process
                if plotimgs:
                    images.append(image[0].squeeze().numpy())
                    images.append(mask[0].squeeze().numpy())
                    images.append(last[0].squeeze().cpu().numpy())
                    images.append(weight[0].squeeze().cpu().numpy()[1, :, :])
                    out = torch.sigmoid(output[0])
                    result_npy = out.max(0)[1].data.squeeze().cpu().numpy()
                    result_npy = np.array(result_npy).astype('uint8') * 255
                    images.append(result_npy)
                    result_npy_nc = morphology.skeletonize(result_npy / 255) * 255  # 骨架化 Skeletonization
                    images.append(result_npy_nc)

                    # [0] original  [1] mask  [2] result [3] result_fc
                    plot_imgs(str(epoch).zfill(3) + '_train', images)
        # Delete a directory
        shutil.rmtree(downsample_folder)

    def test(self, epoch, test_loss_list,temporary_folder):
        self.model.eval()
        test_loss = 0
        images = []
        count = 0
        downsample_folder=os.path.join(temporary_folder,"val","epoch_"+str(int(epoch)))
        os.makedirs(downsample_folder,exist_ok=True)
        F=self.forward_test_loader
        
            
        for batch_idx, (image, mask, last, weight) in enumerate(F):
            
            count += 1
            if torch.cuda.is_available():
                label = mask.squeeze(1).long()  
                img, label, last, weight = image.cuda(), label.cuda(), last.cuda(), weight.cuda()
            # print(batch_idx, ":", image[0].shape)  # Remove parentheses from image[0].shape()
            # normalize
            last[last == 255] = -6
            last[last == 0] = 1

            output, downsample_result = self.model(img, last)

            if self.loss_type == 'abw':
                test_loss += float(self.criterion(output, label, weight, method=self.method).item())
            else:
                test_loss += float(self.criterion(output, label).item())

            if batch_idx == 1:
                #print("\n is in here\n")
                images.append(image[0].squeeze().numpy())
                images.append(mask[0].squeeze().numpy())
                images.append(last[0].squeeze().cpu().numpy())
                images.append(weight[0].squeeze().cpu().numpy()[1, :, :])
                out = torch.sigmoid(output)
                result_npy = out.max(1)[1].data.squeeze().cpu().numpy()
                result_npy = np.array(result_npy).astype('uint8') * 255
                images.append(result_npy[0])
                result_npy_nc = morphology.skeletonize(result_npy / 255) * 255  # 骨架化
                images.append(result_npy_nc[0])
        test_loss /= int(count)
        test_loss_list.append(test_loss)
        return test_loss, images
        # Delete a directory
        shutil.rmtree(downsample_folder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WPU-Net training')
    parser.add_argument('--input', type=str, default=os.path.join(cwd, "/", "datasets", "segmentation", "net_train","/"), help="dataset dir")
    parser.add_argument("--bs", type=int, default=24, help="batch size")
    parser.add_argument("--loss", type=str, default="abw", help="abw/cbw")
    parser.add_argument('--epochs', type=int, default=500, help="epochs")
    parser.add_argument('--ml', help='apply multi_layer', action='store_true')
    parser.add_argument('--two_way_propagation',  default='False')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(args.input)
    loss_list = []
    test_loss_list = []
    test_baseline = 10
    st_time = time.time()
    # model save address
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    checkpoint_path = os.path.join(cwd, "WPU_Net_model/",current_time)
    os.makedirs(checkpoint_path,exist_ok=True)
    trainer= Trainer(bs=args.bs, loss_type=args.loss, multi_layer=args.ml,  data_dir=args.input, workers=args.workers)
    temporary_folder=os.path.join(cwd,"temporary_data")
    for i in range(1, args.epochs + 1):
        # i==current epoch number
        print("currently at epoch:",i)
        if (i - 1) % 10 == 0:
            plotimgs = True
        else:
            plotimgs = False
        trainer.train(i, loss_list, plotimgs, temporary_folder)

        test_loss, imgs = trainer.test(i,test_loss_list, temporary_folder)
        
        f = open(checkpoint_path + 'loss.txt', 'a')
        f.write("epoch %d:loss %f \n" % (i, loss_list[-1]))
        f.write("epoch %d:loss %f test_loss %f\n" % (i, loss_list[-1], test_loss_list[-1]))
        f.close()
        plot(i, loss_list, test_loss_list)
        if (i - 1) % 10 == 0:
            plot_imgs(str(i).zfill(3) + '_test', imgs)
            torch.save(trainer.model.state_dict(), checkpoint_path + str(i) + '_epoch.pth')
        if test_loss < test_baseline:
            test_baseline = test_loss
            torch.save(trainer.model.state_dict(), checkpoint_path + "_best_model_state.pth")
            best_model_num=int(i-1)
        print("final loss:", loss_list[-1],"\nbest model loss:",loss_list[best_model_num])

    ed_time = time.time()
    print('end training , it costs ', str(ed_time-st_time), ' sec in total...')



    # parser = argparse.ArgumentParser(description='WPU-Net training')
    # parser.add_argument('--input', type=str, default=os.path.join(cwd, "/", "datasets", "segmentation", "net_train","/"), help="dataset dir")
    # parser.add_argument("--bs", type=int, default=24, help="batch size")
    # parser.add_argument("--loss", type=str, default="abw", help="abw/cbw")
    # parser.add_argument('--epochs', type=int, default=500, help="epochs")
    # parser.add_argument('--ml', help='apply multi_layer', action='store_true')
    # args = parser.parse_args()
    # print("args",args.input)
    # loss_list = []
    # test_loss_list = []
    # test_baseline = 10000
    # st_time = time.time()
    # # model save address
    # print("cwd",cwd)
    # checkpoint_path = os.path.join(cwd, "WPU_Net_model/")
    # os.makedirs(checkpoint_path,exist_ok=True)
    # print("checkpoint_path:",checkpoint_path)
    # trainer = Trainer(bs=args.bs, loss_type=args.loss, multi_layer=args.ml, data_dir=args.input)
    # for i in range(1, args.epochs + 1):
    #     print("currently at epoch:",i)
    #     if (i - 1) % 10 == 0:
    #         plotimgs = True
    #     else:
    #         plotimgs = False
    #     trainer.train(i, loss_list, plotimgs)
    #     test_loss, imgs = trainer.test(test_loss_list)
    #     f = open(checkpoint_path + 'loss.txt', 'a')
    #     f.write("epoch %d:loss %f \n" % (i, loss_list[-1]))
    #     f.write("epoch %d:loss %f test_loss %f\n" % (i, loss_list[-1], test_loss_list[-1]))
    #     f.close()
    #     plot(i, loss_list, test_loss_list)
    #     if (i - 1) % 10 == 0:
    #         plot_imgs(str(i).zfill(3) + '_test', imgs)
    #         torch.save(trainer.model.state_dict(), checkpoint_path + str(i) + '_epoch.pth')
    #     if test_loss < test_baseline:
    #         test_baseline = test_loss
    #         torch.save(trainer.model.state_dict(), checkpoint_path + "_best_model_state.pth")
    #     print("loss:", loss_list[-1])

    # ed_time = time.time()
    # print('end training , it costs ', str(ed_time-st_time), ' sec in total...')





