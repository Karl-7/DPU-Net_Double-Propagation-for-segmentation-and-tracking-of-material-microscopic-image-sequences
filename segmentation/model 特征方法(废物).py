# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


#此文件为单向传播的模型文件



# Pytorch implementation of "U-net: Convolutional networks for biomedical image segmentation."
# This code is based on  https://github.com/shreyaspadhy/UNet-Zoo/blob/master/models.py
class UNet(nn.Module):
    def __init__(self, num_channels=1,num_classes=2,multi_layer=True):
        super(UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    Conv3x3(num_feat[3], num_feat[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1))

    def forward(self, inputs,last=None,opposite_result=None):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        down4_feat = self.down4(down3_feat)
        bottom_feat = self.bottom(down4_feat)

        up1_feat = self.up1(bottom_feat, down4_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down3_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down2_feat)
        up3_feat = self.upconv3(up3_feat)
        up4_feat = self.up4(up3_feat, down1_feat)
        up4_feat = self.upconv4(up4_feat)

        outputs = self.final(up4_feat)

        return outputs


# WPU-Net  Pytorch implementation of our paper. It is implemented based on the U-Net code above.
class WPU_Net(nn.Module):
    def __init__(self, num_channels=1, num_classes=2, multi_layer=True):
        super(WPU_Net, self).__init__()
        num_feat = [64, 128, 256, 512, 1024,2048]
        self.multi_layer = multi_layer
        print('multi_layer ', self.multi_layer)

        self.down1 = Conv3x3(num_channels, num_feat[0])

        if self.multi_layer:
            addition = 1
        else:
            addition = 0

        self.down2 = Conv3x3(num_feat[0] + addition, num_feat[1])

        self.down3 = Conv3x3(num_feat[1] + addition, num_feat[2])

        self.down4 = Conv3x3(num_feat[2] + addition, num_feat[3])

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.larger_scale_up1=UpConcat(num_feat[5], num_feat[3])
        #特征融合网络（Feature Fusion Network）
        self.fusion_layer = nn.Conv2d(num_feat[5], num_feat[4], kernel_size=1)
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1))

        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2))

    def forward(self, inputs, last, opposite_result=None):
        # print('input for training',inputs,'\n')
        # print('last for training',last,'\n')
        # Multi-level fusion
        # Concatenate the input and last feature maps along the channel dimension
        inputs = torch.cat([inputs, last], 1)
        # Pass the concatenated input through the first downsample block
        down1_feat = self.down1(inputs)

        if self.multi_layer:
            # Downsample the last feature map using max pooling
            down2_last = self.pool(last)
            # Concatenate the downsampled last feature map and the downsampled down1 feature map
            down2_pool = torch.cat([self.pool(down1_feat), down2_last], 1)
            # Pass the concatenated feature map through the second downsample block
            down2_feat = self.down2(down2_pool)

            down3_last = self.pool(down2_last)
            down3_pool = torch.cat([self.pool(down2_feat), down3_last], 1)
            down3_feat = self.down3(down3_pool)

            down4_last = self.pool(down3_last)
            down4_pool = torch.cat([self.pool(down3_feat), down4_last], 1)
            down4_feat = self.down4(down4_pool)
        else:
            # Downsample the down1 feature map using max pooling
            down2_feat = self.down2(self.pool(down1_feat))
            down3_feat = self.down3(self.pool(down2_feat))
            down4_feat = self.down4(self.pool(down3_feat))

        # Pass the downsampled feature map through the bottom block
        if opposite_result==None:
            bottom_feat = self.bottom(down4_feat)
            swiftup=self.up1
            
            # Pass the bottom feature map and the corresponding downsampled feature map through the first upsample block
            up1_feat = swiftup(bottom_feat, down4_feat)
            up1_feat = self.upconv1(up1_feat)
            # Pass the upsampled feature map and the corresponding downsampled feature map through the second upsample block
            up2_feat = self.up2(up1_feat, down3_feat)
            up2_feat = self.upconv2(up2_feat)
            # Pass the upsampled feature map and the corresponding downsampled feature map through the third upsample block
            up3_feat = self.up3(up2_feat, down2_feat)
            up3_feat = self.upconv3(up3_feat)
            # Pass the upsampled feature map and the corresponding downsampled feature map through the fourth upsample block
            up4_feat = self.up4(up3_feat, down1_feat)
            up4_feat = self.upconv4(up4_feat)

            outputs = self.final(up4_feat)
            return outputs, bottom_feat
        else:
            bottom_feature=[]
            #特征拼接：
            bottom_feature.append(torch.cat([self.bottom(down4_feat), opposite_result], 1))
            swiftup=self.larger_scale_up1
            #特征相乘
            bottom_feature.append(self.bottom(down4_feat) * opposite_result)
            #特征相加
            bottom_feature.append(self.bottom(down4_feat) + opposite_result)
            #特征乘法（Feature Multiplication）
            bottom_feature.append(self.fusion_layer(torch.cat([self.bottom(down4_feat), opposite_result], 1)))
            outputs=[]
            
            for x in range (4):
                bottom_feat = bottom_feature[x]
                if x==0:
                    swiftup=self.larger_scale_up1
                else:
                    swiftup=self.up1
                # Pass the bottom feature map and the corresponding downsampled feature map through the first upsample block
                up1_feat = swiftup(bottom_feat, down4_feat)
                up1_feat = self.upconv1(up1_feat)
                # Pass the upsampled feature map and the corresponding downsampled feature map through the second upsample block
                up2_feat = self.up2(up1_feat, down3_feat)
                up2_feat = self.upconv2(up2_feat)
                # Pass the upsampled feature map and the corresponding downsampled feature map through the third upsample block
                up3_feat = self.up3(up2_feat, down2_feat)
                up3_feat = self.upconv3(up3_feat)
                # Pass the upsampled feature map and the corresponding downsampled feature map through the fourth upsample block
                up4_feat = self.up4(up3_feat, down1_feat)
                up4_feat = self.upconv4(up4_feat)
                # outputs.append(self.final(up4_feat))
                outputs.append(self.final(up4_feat))
            print("output.shape:",len(outputs))
            return outputs
class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out

