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
import numpy as np
import os
import math
from PIL import Image
import random
import cv2

# generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    # TODO:
    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # the second dimension 4 means each cell has 4 default bounding boxes.
    # their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    # where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    # for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    # the last dimension 8 means each default bounding box has 8 attributes:
    # [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    grid = 0
    for layer in layers:
        grid += layer ** 2
    boxes = np.zeros([grid, 4, 8])
    counter = 0
    for i in range(len(layers)):
        grid_num = layers[i]
        ssize = small_scale[i]
        lsize = large_scale[i]
        lsize_mul = lsize * math.sqrt(2)
        lsize_div = lsize / math.sqrt(2)
        llsize_mul = round(lsize_mul, 2)
        llsize_div = round(lsize_div, 2)
        if i > 0:
            counter += layers[i-1]**2
        for j in range(layers[i]):
            for k in range(layers[i]):
                x_center = (0.5 + j) / grid_num
                x_center = round(x_center, 2)
                y_center = (0.5 + k) / grid_num
                y_center = round(y_center, 2)
                # box1
                x_min = x_center - ssize / 2
                x_min = round(x_min, 2) if x_min > 0 else 0
                y_min = y_center - ssize / 2
                y_min = round(y_min, 2) if y_min > 0 else 0
                x_max = x_center + ssize / 2
                x_max = round(x_max, 2) if x_max < 1 else 1
                y_max = y_center + ssize / 2
                y_max = round(y_max, 2) if y_max < 1 else 1
                boxes[counter + j*grid_num+k, 0] = [x_center, y_center, ssize, ssize, x_min, y_min, x_max, y_max]
                # boxes.append([x_center, y_center, ssize, ssize, x_min, y_min, x_max, y_max])
                # box2
                x_min = x_center - lsize / 2
                x_min = round(x_min, 2) if x_min > 0 else 0
                y_min = y_center - lsize / 2
                y_min = round(y_min, 2) if y_min > 0 else 0
                x_max = x_center + lsize / 2
                x_max = round(x_max, 2) if x_max < 1 else 1
                y_max = y_center + lsize / 2
                y_max = round(y_max, 2) if y_max < 1 else 1
                boxes[counter + j * grid_num + k, 1] = [x_center, y_center, lsize, lsize, x_min, y_min, x_max, y_max]
                # boxes.append([x_center, y_center, lsize, lsize, x_min, y_min, x_max, y_max])
                # box3
                x_min = x_center - lsize_mul / 2
                x_min = round(x_min, 2) if x_min > 0 else 0
                y_min = y_center - lsize_div / 2
                y_min = round(y_min, 2) if y_min > 0 else 0
                x_max = x_center + lsize_mul / 2
                x_max = round(x_max, 2) if x_max < 1 else 1
                y_max = y_center + lsize_div / 2
                y_max = round(y_max, 2) if y_max < 1 else 1
                boxes[counter + j * grid_num + k, 2] = [x_center, y_center, llsize_mul, llsize_div, x_min, y_min, x_max, y_max]
                # boxes.append([x_center, y_center, llsize_mul, llsize_div, x_min, y_min, x_max, y_max])
                # box4
                x_min = x_center - lsize_div / 2
                x_min = round(x_min, 2) if x_min > 0 else 0
                y_min = y_center - lsize_mul / 2
                y_min = round(y_min, 2) if y_min > 0 else 0
                x_max = x_center + lsize_div / 2
                x_max = round(x_max, 2) if x_max < 1 else 1
                y_max = y_center + lsize_mul / 2
                y_max = round(y_max, 2) if y_max < 1 else 1
                boxes[counter + j * grid_num + k, 3] = [x_center, y_center, llsize_div, llsize_mul, x_min, y_min, x_max, y_max]
                # boxes.append([x_center, y_center, llsize_div, llsize_mul, x_min, y_min, x_max, y_max])
    boxes = boxes.reshape([grid*4, 8])
    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...],
    # where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)
    
    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:, 6]-boxs_default[:, 4])*(boxs_default[:, 7]-boxs_default[:, 5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold,
    #                            then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box
    
    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    gx = (x_max + x_min)/2
    gy = (y_max + y_min)/2
    gw = x_max - x_min
    gh = y_max - y_min
    
    ious_true = ious > threshold
    # TODO:
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold,
    # then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    for idx, istrue in enumerate(ious_true):
        if istrue:
            # px = boxs_default[idx, 0]
            # py = boxs_default[idx, 1]
            # pw = boxs_default[idx, 2]
            # ph = boxs_default[idx, 3]
            tx = (gx - boxs_default[idx, 0]) / boxs_default[idx, 2]
            ty = (gy - boxs_default[idx, 1]) / boxs_default[idx, 3]
            tw = math.log(gw / boxs_default[idx, 2])
            th = math.log(gh / boxs_default[idx, 3])
            ann_box[idx] = [tx, ty, tw, th]
            ann_confidence[idx, cat_id] = 1
            ann_confidence[idx, 3] = 0

    ious_true = np.argmax(ious)
    # TODO:
    # make sure at least one default bounding box is used
    # update ann_box and ann_confidence (do the same thing as above)
    tx = (gx - boxs_default[ious_true, 0]) / boxs_default[ious_true, 2]
    ty = (gy - boxs_default[ious_true, 1]) / boxs_default[ious_true, 3]
    tw = math.log(gw / boxs_default[ious_true, 2])
    th = math.log(gh / boxs_default[ious_true, 3])
    ann_box[ious_true] = [tx, ty, tw, th]
    ann_confidence[ious_true, cat_id] = 1
    ann_confidence[ious_true, 3] = 0


# mode: Train, Test. Eval, Visual
class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, image_size=320, mode='Train'):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        self.mode = mode
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        train_cut = math.ceil(len(self.img_names)*0.9)
        if self.train:
            self.img_names = self.img_names[:train_cut]
        elif self.mode == 'Eval':
            self.img_names = self.img_names[train_cut:]

        # self.trans = transforms.PILToTensor()
        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background
        
        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"
        
        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3]+"txt"

        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        # 4. Data augmentation. You need to implement random cropping first.
        # You can try adding other augmentations to get better results.
        
        # to use function "match":
        # match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box,
        # normalized with respect to the width or height of the image.
        
        # note:
        # please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in a image with (width=1000, height=500)
        # will be normalized to (x/width=0.1,y/height=0.4)

        # read image
        image = Image.open(img_name)
        width, height = image.size

        # adjust gray scala image to 3 channel
        if len(image.split()) == 1:
            image = image.convert('RGB')

        # read annotation
        ann_list = []
        if self.mode != 'Visual':
            with open(ann_name, 'r') as f:
                for line in f:
                    c_id, x_min, y_min, w, h = line.split()
                    x_max = float(x_min) + float(w)
                    y_max = float(y_min) + float(h)
                    ann_list.append([int(c_id), float(x_min), float(y_min), x_max, y_max])

        ann_list = np.array(ann_list)

        # TODO:
        # augmentation
        if self.mode != 'Test' and self.mode != 'Visual':
            # crop_xmin = random.randint(0, math.floor(np.min(ann_list[:, 1], axis=0)))
            # crop_ymin = random.randint(0, math.floor(np.min(ann_list[:, 2], axis=0)))
            # crop_xmax = random.randint(math.ceil(np.max(ann_list[:, 3])), width)
            # crop_ymax = random.randint(math.ceil(np.max(ann_list[:, 4])), height)

            crop_xmin = random.randint(0, min(math.floor(np.min(ann_list[:, 1], axis=0)), 10))
            crop_ymin = random.randint(0, min(math.floor(np.min(ann_list[:, 2], axis=0)), 10))
            crop_xmax = random.randint(max(math.ceil(np.max(ann_list[:, 3])), width-10), width)
            crop_ymax = random.randint(max(math.ceil(np.max(ann_list[:, 4])), height-10), height)

            image = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            width_new, height_new = image.size
            image = image.resize((self.image_size, self.image_size))

            ann_list[:, 1] = (ann_list[:, 1] - crop_xmin) / width_new
            ann_list[:, 3] = (ann_list[:, 3] - crop_xmin) / width_new
            ann_list[:, 2] = (ann_list[:, 2] - crop_ymin) / height_new
            ann_list[:, 4] = (ann_list[:, 4] - crop_ymin) / height_new
        elif self.mode == 'Test':
            image = image.resize((self.image_size, self.image_size))
            ann_list[:, 1] = ann_list[:, 1] / width
            ann_list[:, 3] = ann_list[:, 3] / width
            ann_list[:, 2] = ann_list[:, 2] / height
            ann_list[:, 4] = ann_list[:, 4] / height
        else:
            image = image.resize((self.image_size, self.image_size))

        image = self.trans(image)

        # match
        if self.mode != 'Visual':
            # ann = [class_id,x_min,y_min,x_max,y_max]
            for ann in ann_list:
                match(ann_box, ann_confidence, self.boxs_default, self.threshold, int(ann[0]), ann[1], ann[2], ann[3], ann[4])

        if self.mode == 'Visual':
            ann_box = np.array([])
            ann_confidence = np.array([])

        return image, ann_box, ann_confidence, self.img_names[index]
