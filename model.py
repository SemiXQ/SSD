import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box, alpha=1):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

    class_num = pred_confidence.size(dim=2)

    pred_confidence = torch.reshape(pred_confidence, (-1, class_num))
    pred_box = torch.reshape(pred_box, (-1, 4))
    ann_confidence = torch.reshape(ann_confidence, (-1, class_num))
    ann_box = torch.reshape(ann_box, (-1, 4))

    nonzero_idx, _ = torch.nonzero(ann_confidence[:, :3], as_tuple=True)
    # print(nonzero_idx.shape)
    zero_idx = (ann_confidence[:, 3] == torch.tensor([1]).cuda()).nonzero(as_tuple=True)[0]

    pred_cfd = torch.index_select(pred_confidence, 0, nonzero_idx)
    anno_cfd = torch.index_select(ann_confidence, 0, nonzero_idx)
    pred_cfd_non_box = torch.index_select(pred_confidence, 0, zero_idx)
    anno_cfd_non_box = torch.index_select(ann_confidence, 0, zero_idx)
    pred_b = torch.index_select(pred_box, 0, nonzero_idx)
    anno_b = torch.index_select(ann_box, 0, nonzero_idx)

    loss_cls = F.cross_entropy(pred_cfd, anno_cfd) + 3 * F.cross_entropy(pred_cfd_non_box, anno_cfd_non_box)
    loss_box = F.smooth_l1_loss(pred_b, anno_b)
    loss = loss_cls + alpha * loss_box

    return loss


class Conv2dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Conv2dBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class BeginBlock(nn.Module):
    def __init__(self):
        super(BeginBlock, self).__init__()
        self.layer1 = Conv2dBlock(in_channel=3, out_channel=64, kernel=3, stride=2, padding=1)
        self.layer2 = Conv2dBlock(in_channel=64, out_channel=64, kernel=3, stride=1, padding=1)
        self.layer3 = Conv2dBlock(in_channel=64, out_channel=64, kernel=3, stride=1, padding=1)
        self.layer4 = Conv2dBlock(in_channel=64, out_channel=128, kernel=3, stride=2, padding=1)
        self.layer5 = Conv2dBlock(in_channel=128, out_channel=128, kernel=3, stride=1, padding=1)
        self.layer6 = Conv2dBlock(in_channel=128, out_channel=128, kernel=3, stride=1, padding=1)
        self.layer7 = Conv2dBlock(in_channel=128, out_channel=256, kernel=3, stride=2, padding=1)
        self.layer8 = Conv2dBlock(in_channel=256, out_channel=256, kernel=3, stride=1, padding=1)
        self.layer9 = Conv2dBlock(in_channel=256, out_channel=256, kernel=3, stride=1, padding=1)
        self.layer10 = Conv2dBlock(in_channel=256, out_channel=512, kernel=3, stride=2, padding=1)
        self.layer11 = Conv2dBlock(in_channel=512, out_channel=512, kernel=3, stride=1, padding=1)
        self.layer12 = Conv2dBlock(in_channel=512, out_channel=512, kernel=3, stride=1, padding=1)
        self.layer13 = Conv2dBlock(in_channel=512, out_channel=256, kernel=3, stride=2, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        return x


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num # num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.beginConv = BeginBlock()
        ##### right
        self.block_red_1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 100)
        self.block_blue_1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 100)

        self.block_1_a = Conv2dBlock(in_channel=256, out_channel=256, kernel=1, stride=1, padding=0)
        self.block_1_b = Conv2dBlock(in_channel=256, out_channel=256, kernel=3, stride=2, padding=1)
        ##### right
        self.block_red_2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 25)
        self.block_blue_2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 25)


        self.block_11_a = Conv2dBlock(in_channel=256, out_channel=256, kernel=1, stride=1, padding=0)
        self.block_11_b = Conv2dBlock(in_channel=256, out_channel=256, kernel=3, stride=1, padding=0)
        ##### right
        self.block_red_3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 9)
        self.block_blue_3 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1)
        # reshape (N, 16, 9)


        self.block_111_a = Conv2dBlock(in_channel=256, out_channel=256, kernel=1, stride=1, padding=0)
        self.block_111_b = Conv2dBlock(in_channel=256, out_channel=256, kernel=3, stride=1, padding=0)
        ##### left
        self.block_red_4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0)
        # reshape (N, 16, 1)
        self.block_blue_4 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0)
        # reshape (N, 16, 1)

        self.softmax = nn.Softmax(dim=2)
        # Since F.CrossEntropyLoss includes a calculation of LogSoftmax, I will use softmax in main after loss calculate


    def forward(self, x):
        #input:
        # x -- images, [batch_size, 3, 320, 320]

        # x = x/255.0 # normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        # TODO: define forward
        
        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.)
        # If yes, which dimension should you apply softmax?
        
        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        batch_size = x.size(dim=0)
        x = self.beginConv(x)
        box1 = self.block_red_1(x)
        box1 = torch.reshape(box1, (batch_size,16,-1))
        cfd1 = self.block_blue_1(x)
        cfd1 = torch.reshape(cfd1, (batch_size, 16, -1))

        x = self.block_1_a(x)
        x = self.block_1_b(x)
        box2 = self.block_red_2(x)
        box2 = torch.reshape(box2, (batch_size, 16, -1))
        cfd2 = self.block_blue_2(x)
        cfd2 = torch.reshape(cfd2, (batch_size, 16, -1))

        x = self.block_11_a(x)
        x = self.block_11_b(x)
        box3 = self.block_red_3(x)
        box3 = torch.reshape(box3, (batch_size, 16, -1))
        cfd3 = self.block_blue_3(x)
        cfd3 = torch.reshape(cfd3, (batch_size, 16, -1))

        x = self.block_111_a(x)
        x = self.block_111_b(x)
        box4 = self.block_red_4(x)
        box4 = torch.reshape(box4, (batch_size, 16, -1))
        cfd4 = self.block_blue_4(x)
        cfd4 = torch.reshape(cfd4, (batch_size, 16, -1))

        bboxes = torch.cat((box1, box2, box3, box4), dim=2)
        confidence = torch.cat((cfd1, cfd2, cfd3, cfd4), dim=2)

        bboxes = torch.permute(bboxes, (0, 2, 1))
        confidence = torch.permute(confidence, (0, 2, 1))

        bboxes = torch.reshape(bboxes, (batch_size, -1, 4))
        confidence = torch.reshape(confidence, (batch_size, -1, self.class_num))

        confidence = self.softmax(confidence)
        # Since F.CrossEntropyLoss includes a calculation of LogSoftmax, I will use softmax in main after loss calculate

        return confidence, bboxes
