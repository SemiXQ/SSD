import numpy as np
import cv2
from dataset import iou
import math


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred_only(windowname, pred_confidence, pred_box, image_, boxs_default, image_names_):
    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)

    image = image_ * 255
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

    # image3 = np.zeros(image.shape, np.uint8)
    # image4 = np.zeros(image.shape, np.uint8)
    # image3[:] = image[:]
    # image4[:] = image[:]

    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    width, height, _ = image.shape

    # draw pred
    ''''''
    pred_confidence_, pred_box_, default_box_used = non_maximum_suppression(pred_confidence, pred_box, boxs_default,
                                                                            overlap=0.3, threshold=0.6)

    for i in range(len(pred_confidence_)):
        for j in range(class_num):
            if pred_confidence_[i, j] > 0.7:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                # image4: draw network-predicted "default" boxes on image4
                # (to show which cell does your network think that contains an object)

                # predicted bounding box
                xmin = int(max(round(pred_box_[i, 4] * width), 0))
                ymin = int(max(round(pred_box_[i, 5] * height), 0))
                xmax = int(min(round(pred_box_[i, 6] * width), width))
                ymax = int(min(round(pred_box_[i, 7] * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)

                # default box used
                xmin = int(max(round(default_box_used[i, 4] * width), 0))
                ymin = int(max(round(default_box_used[i, 5] * height), 0))
                xmax = int(min(round(default_box_used[i, 6] * width), width))
                ymax = int(min(round(default_box_used[i, 7] * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image4, start_point, end_point, color, thickness)

    # combine four images into one
    # image = np.zeros([height, width * 2, 3], np.uint8)
    # image[:, :width] = image3
    # image[:, width:] = image4

    image = np.zeros([height * 2, width * 2, 3], np.uint8)
    image[:height, :width] = image1
    image[:height, width:] = image2
    image[height:, :width] = image3
    image[height:, width:] = image4

    # output_dir = 'visual_output/' + image_names_
    output_dir = 'output/' + windowname + '/' + image_names_
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    cv2.imwrite(output_dir, image)
    # cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization


def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, image_names_,
                   do_nms=True, nms_threshold=0.6, nms_overlap=0.3):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    # image_names_    -- image name for saving
    # do_nms          -- whether to use nms, boolean, default: True
    # nms_threshold   -- threshold used for nms
    # nms_overlap     -- overlap used for nms
    
    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num-1
    # class_num = 3 now, because we do not need the last class (background)

    image = image_ * 255
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    width, height, _ = image.shape
    
    # draw ground truth
    for i in range(len(ann_confidence)):
        if len(ann_confidence) < 1:
            break
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2
                # (to show that you have assigned the object to the correct cell/cells)
                
                # you can use cv2.rectangle as follows:
                # start_point = (x1, y1) # top left corner, x1<x2, y1<y2
                # end_point = (x2, y2) # bottom right corner
                # color = colors[j] # use red green blue to represent different classes
                # thickness = 2
                # cv2.rectangle(image?, start_point, end_point, color, thickness)

                # ground truth bounding box
                gx = boxs_default[i, 2] * ann_box[i, 0] + boxs_default[i, 0]
                gy = boxs_default[i, 3] * ann_box[i, 1] + boxs_default[i, 1]
                gw = boxs_default[i, 2] * math.exp(ann_box[i, 2])
                gh = boxs_default[i, 3] * math.exp(ann_box[i, 3])

                xmin = int(max(round((gx - 0.5 * gw) * width), 0))
                ymin = int(max(round((gy - 0.5 * gh) * height), 0))
                xmax = int(min(round((gx + 0.5 * gw) * width), width))
                ymax = int(min(round((gy + 0.5 * gh) * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image1, start_point, end_point, color, thickness)

                # ground truth default box
                xmin = int(max(round(boxs_default[i, 4] * width), 0))
                ymin = int(max(round(boxs_default[i, 5] * height), 0))
                xmax = int(min(round(boxs_default[i, 6] * width), width))
                ymax = int(min(round(boxs_default[i, 7] * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image2, start_point, end_point, color, thickness)

    # pred
    ''''''
    if do_nms:
        pred_confidence_, pred_box_, default_box_used = non_maximum_suppression(pred_confidence, pred_box, boxs_default,
                                                                                overlap=nms_overlap,
                                                                                threshold=nms_threshold)
    else:
        pred_confidence_ = pred_confidence
        pred_box_ = transform_to_truth_location(pred_box, boxs_default)
        default_box_used = boxs_default

    for i in range(len(pred_confidence_)):
        for j in range(class_num):
            if pred_confidence_[i, j] > 0.5:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                # image4: draw network-predicted "default" boxes on image4
                # (to show which cell does your network think that contains an object)

                # predicted bounding box
                xmin = int(max(round(pred_box_[i, 4] * width), 0))
                ymin = int(max(round(pred_box_[i, 5] * height), 0))
                xmax = int(min(round(pred_box_[i, 6] * width), width))
                ymax = int(min(round(pred_box_[i, 7] * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)

                # default box used
                xmin = int(max(round(default_box_used[i, 4] * width), 0))
                ymin = int(max(round(default_box_used[i, 5] * height), 0))
                xmax = int(min(round(default_box_used[i, 6] * width), width))
                ymax = int(min(round(default_box_used[i, 7] * height), height))

                start_point = (xmin, ymin)  # top left corner, x1<x2, y1<y2
                end_point = (xmax, ymax)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image4, start_point, end_point, color, thickness)


    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h*2, w*2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    output_dir = 'output/'+windowname+'/'+image_names_
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    cv2.imwrite(output_dir, image)
    # cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def transform_to_truth_location(box_, boxs_default):
    # box_: x_center, y_center, width, height
    # boxs_default: x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max
    true_box = np.zeros((box_.shape[0], 8), dtype=float)
    for i in range(box_.shape[0]):

        # px = boxs_default[i,0]
        # py = boxs_default[i,1]
        # pw = boxs_default[i,2]
        # ph = boxs_default[i,3]

        # print(boxs_default[i], box_[i])
        gx = boxs_default[i, 2] * box_[i, 0] + boxs_default[i, 0]
        gy = boxs_default[i, 3] * box_[i, 1] + boxs_default[i, 1]
        gw = boxs_default[i, 2] * math.exp(box_[i, 2])
        gh = boxs_default[i, 3] * math.exp(box_[i, 3])

        # true : x_center, y_center, box_width, box_height, xmin, ymin, xmax, ymax
        true_box[i, 0] = round(gx, 2)
        true_box[i, 1] = round(gy, 2)
        true_box[i, 2] = round(gw, 2)
        true_box[i, 3] = round(gh, 2)
        true_box[i, 4] = round(gx - 0.5 * gw, 2)
        true_box[i, 5] = round(gy - 0.5 * gh, 2)
        true_box[i, 6] = round(gx + 0.5 * gw, 2)
        true_box[i, 7] = round(gy + 0.5 * gh, 2)

    return true_box


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]  # x_center, y_center, width, height
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap,
    #                   then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold,
    #                   then consider this cell carrying a bounding box with this class.
    
    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above,
    # you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes,
    # and write a new visualization function for that.
    
    #TODO: non maximum suppression
    pred_box = []
    pred_confidence = []
    default_box_used = []

    pred_box_8 = transform_to_truth_location(box_, boxs_default)
    # pred_box_8 = np.zeros((box_.shape[0], 8), dtype=float)
    # print(pred_box_8.shape)
    # pred_box_8[:, :4] = box_[:, :]
    # print()
    # pred_box_8[:, 4] = box_[:, 0] - 0.5 * box_[:, 2]
    # pred_box_8[:, 5] = box_[:, 1] - 0.5 * box_[:, 3]
    # pred_box_8[:, 6] = box_[:, 0] + 0.5 * box_[:, 2]
    # pred_box_8[:, 7] = box_[:, 1] + 0.5 * box_[:, 3]
    confidence_copy = confidence_.copy()
    dbox_copy = boxs_default.copy()

    while(len(pred_box_8)>0):
        a = len(pred_box_8)
        max_cfd_list = np.max(confidence_copy[:, :3], axis=1)
        max_cfd = np.max(max_cfd_list)
        if max_cfd <= threshold:
            break
        max_idx = np.argmax(max_cfd_list)
        # pred_box.append(true_box[max_idx, :4])
        pred_box.append(pred_box_8[max_idx])
        pred_confidence.append(confidence_copy[max_idx])
        default_box_used.append(dbox_copy[max_idx])

        # print(pred_box_8[max_idx, 4], pred_box_8[max_idx, 5], pred_box_8[max_idx, 6], pred_box_8[max_idx, 7])

        iou_list = iou(pred_box_8, pred_box_8[max_idx, 4], pred_box_8[max_idx, 5], pred_box_8[max_idx, 6], pred_box_8[max_idx, 7])
        del_list = np.where(iou_list > np.array([overlap]))[0]

        pred_box_8 = np.delete(pred_box_8, obj=del_list, axis=0)
        confidence_copy = np.delete(confidence_copy, obj=del_list, axis=0)
        dbox_copy = np.delete(dbox_copy, obj=del_list, axis=0)

    pred_box = np.array(pred_box)
    pred_confidence = np.array(pred_confidence)
    default_box_used = np.array(default_box_used)

    return pred_confidence, pred_box, default_box_used
