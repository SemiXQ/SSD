import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

import math
import matplotlib.pyplot as plt
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False)  # False
# mode = Eval for visualization only
parser.add_argument('--mode', type=str, default='Test')  # 'Test'
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4  # cat dog person background

num_epochs = 100  # 30  # 100
batch_size = 32  # 32
# loss sum weight
alpha = 1
learn_rate = 1e-4

boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

if not args.test:
    train_loss = []
    val_loss = []

    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True,
                   image_size=320, mode='Train')
    dataset_eval = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False,
                        image_size=320, mode='Eval')

    # dataset = COCO("temp/train/images/", "temp/train/annotations/", class_num, boxs_default, train=True,
    #                image_size=320, mode='Train')
    # dataset_eval = COCO("temp/train/images/", "temp/train/annotations/", class_num, boxs_default, train=False,
    #                     image_size=320, mode='Eval')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # optimizer = optim.Adam(network.parameters(), lr=learn_rate)
    # optimizer = optim.SGD(network.parameters(), lr=learn_rate)
    optimizer = optim.RMSprop(network.parameters(), lr=learn_rate, alpha=0.9)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, image_names_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box, alpha)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

            # pred_confidence = F.softmax(pred_confidence, dim=2)

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        train_loss_eps = avg_loss.detach().cpu().numpy()/avg_count
        # print(type(train_loss_eps), train_loss_eps)
        # pdb.set_trace()
        train_loss.append(train_loss_eps)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        # I have done non-maximum suppression in visualize_pred()

        if epoch % 10 == 9:
            visualize_pred("Train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
                            images_[0].numpy(), boxs_default, image_names_[0])

            # pred_confidence_, pred_box_, _ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
            print(pred_box_.shape)


        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_names_[0])

        ''''''
        #VALIDATION
        network.eval()

        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        # I have done this part in dataset.py

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader_eval, 0):
            images_, ann_box_, ann_confidence_, image_names_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)

            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box, alpha)

            # pred_confidence = F.softmax(pred_confidence, dim=2)

            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()

            avg_loss += loss_net.data
            avg_count += 1

            # optional: implement a function to accumulate precision and recall to compute mAP or F1.
            # update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)

        print('[%d] time: %f val loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        val_loss_eps = avg_loss.detach().cpu().numpy()/avg_count
        # print(type(val_loss_eps), val_loss_eps)
        # pdb.set_trace()
        val_loss.append(val_loss_eps)

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        if epoch % 10 == 9:
            visualize_pred("Eval", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
                           images_[0].numpy(), boxs_default, image_names_[0])

            pred_confidence_, pred_box_, _ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
            print(pred_box_.shape)
        # visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_names_[0])

        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)

        #save weights
        if epoch % 10 == 9:
            # save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')

    lossfig, lossaxes = plt.subplots()
    epoch_list = list(range(num_epochs))
    # print(len(train_loss), train_loss)
    # print(len(val_loss), val_loss)
    # pdb.set_trace()
    lossaxes.plot(epoch_list, train_loss, label='train_loss', color='b')
    lossaxes.plot(epoch_list, val_loss, label='val_loss', color='r')
    lossaxes.set_xlabel('epoch')
    lossaxes.set_ylabel('loss')
    lossaxes.set_title('train_val_loss_plot')
    lossaxes.legend()
    lossfig.show()
    lossfig.savefig('train val loss.png')

elif args.mode == 'Test':
    #TEST
    # dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train=False,
    #                     image_size=320, mode='Test')
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False,
                        image_size=320, mode='Eval')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_names_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        # pred_confidence = F.softmax(pred_confidence, dim=2)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        # pred_confidence_,pred_box_, _ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
        # pred_box_ = pred_box_[:, :4]
        
        # TODO: save predicted bounding boxes and classes to a txt file.
        # you will need to submit those files for grading this assignment
        
        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_names_[0])
        # cv2.waitKey(1000)
        visualize_pred("Test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
                       images_[0].numpy(), boxs_default, image_names_[0], do_nms=True, nms_threshold=0.6,
                       nms_overlap=0.3)

if args.test and args.mode == 'Eval':
    # VISUALIZATION ONLY
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train=False,
                        image_size=320, mode='Visual')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_names_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        # pred_confidence = F.softmax(pred_confidence, dim=2)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        # pred_confidence_,pred_box_, _ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
        # pred_box_ = pred_box_[:, :4]

        # TODO: save predicted bounding boxes and classes to a txt file.
        # you will need to submit those files for grading this assignment

        # visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_names_[0])
        # cv2.waitKey(1000)
        # visualize_pred_only('Visual', pred_confidence_, pred_box_, images_[0].numpy(), boxs_default, image_names_[0])
        visualize_pred('Visual', pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(),
                       images_[0].numpy(), boxs_default, image_names_[0], do_nms=True, nms_threshold=0.4,
                       nms_overlap=0.5)

if args.test and args.mode == 'Print':
    # Print annotation files
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False,
                        image_size=320, mode='Eval')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_names_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        # pred_confidence = F.softmax(pred_confidence, dim=2)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        pred_confidence, pred_box, default_box_used = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default,
                                                                              overlap=0.3, threshold=0.6)

        output_annotations(pred_confidence, pred_box, images_[0].numpy(), image_names_[0], out_mode='Eval/')
