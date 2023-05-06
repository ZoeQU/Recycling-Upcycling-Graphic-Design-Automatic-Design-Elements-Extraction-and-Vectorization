# -*- coding:utf-8 -*-
import numpy as np
import os
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


from utils.data_process.parse_ElBa_jason import GetJsonInfo, get_img_info


def show_segment(gemfield_polygons):
    fig, ax = plt.subplots()
    polygons = []
    gemfield_polygon = gemfield_polygons[0]
    max_value = max(gemfield_polygon) * 1.3
    gemfield_polygon = [i * 1.0/max_value for i in gemfield_polygon]
    poly = np.array(gemfield_polygon).reshape((int(len(gemfield_polygon)/2), 2))
    polygons.append(Polygon(poly, True))
    p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=1)
    colors = 100 * np.random.rand(1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.show()
    plt.close('all')


def eval_segmentation(pred_segments, gt_segments):
    """ Evaluate stored predictions from a segmentation network.
        The semantic masks from the validation dataset are not supposed to change.    """
    #TODO some problems here
    n_classes = len(gt_segments)

    iou_segs = [0] * n_classes
    for i, gt_segment in enumerate(gt_segments):
        # creat gt_mask
        img_template_gt = np.zeros((1024, 1024, 3), np.uint8)
        poly_gt = np.array(gt_segment[0]).reshape((int(len(gt_segment[0]) / 2), 2)).astype(np.uint64)
        cv2.fillPoly(img_template_gt, pts=[poly_gt], color=(255, 255, 255))
        gray_gt = cv2.cvtColor(img_template_gt, cv2.COLOR_BGR2GRAY)
        ret, gt_mask = cv2.threshold(gray_gt, 0, 1, cv2.THRESH_BINARY)  # bk:0, object:1

        temp_acc = []
        for j, pre_seg in enumerate(pred_segments):
            # create pre seg mask
            img_template_pre = np.zeros((1024, 1024, 3), np.uint8)
            cv2.fillPoly(img_template_pre, pts=[pre_seg], color=(255, 255, 255))
            gray_pre = cv2.cvtColor(img_template_pre, cv2.COLOR_BGR2GRAY)
            ret, pre_mask = cv2.threshold(gray_pre, 0, 1, cv2.THRESH_BINARY)
            # TP, FP, and FN evaluation
            tmp_gt = (gt_mask == 1)
            tmp_pred = (pre_mask == 1)
            pre_tp = np.sum(tmp_gt & tmp_pred)
            pre_fp = np.sum(~tmp_gt & tmp_pred)
            pre_fn = np.sum(tmp_gt & ~tmp_pred)
            pre_seg_acc = float(pre_tp) / max(float(pre_tp + pre_fp + pre_fn), 1e-8)
            temp_acc.append(pre_seg_acc)

        if len(temp_acc) > 0:
            pre_acc = max(temp_acc)
        else:
            pre_acc = 0
        iou_segs[i] += pre_acc

    # Write results
    eval_result = dict()
    eval_result['iou_all_categs'] = iou_segs
    eval_result['mIoU'] = np.mean(iou_segs)
    return eval_result


def get_ElBa_selected_info(file_name):
    parser = GetJsonInfo(file_name)
    imgs_info, annotations = parser.read_json()
    return imgs_info, annotations


def get_gt_bkmask(imgs_segs):
    img_template = np.zeros((1024, 1024, 3), np.uint8)
    for seg in imgs_segs:
        poly = np.array(seg[0]).reshape((int(len(seg[0]) / 2), 2)).astype(np.uint64)
        cv2.fillPoly(img_template, pts=[poly], color=(255, 255, 255))
    gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    gt_bk_mask = 1 - thresh
    return gt_bk_mask


def eval_bkmask(pre_bk, img_name, imgs_segs):
    pre_bk = np.array(pre_bk)
    gt_bk = get_gt_bkmask(imgs_segs)

    # TP, FP, and FN evaluation
    tmp_gt = (gt_bk == 1)
    tmp_pred = (pre_bk == 255)
    tp = np.sum(tmp_gt & tmp_pred)
    fp = np.sum(~tmp_gt & tmp_pred)
    fn = np.sum(tmp_gt & ~tmp_pred)
    iou = float(tp) / max(float(tp + fp + fn), 1e-8)
    return iou


def bbox_iou(box1, box2):
    if box1[0] < box2[0] and box1[1] < box2[1]:
        pass
    else:
        temp = box1
        box1 = box2
        box2 = temp
    if box1[0] < box2[0] and box1[1] < box2[1]:
        if box2[0] < box1[2] and box2[1] < box1[3]:
            if box2[0] > box1[0]:  #包含
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                iou = float(min(area1, area2) / max(area1, area2))
            else:  # 只相交 不包含
                bxmin = min(box1[0], box2[0])
                bymin = min(box1[1], box2[1])
                bxmax = max(box1[2], box2[2])
                bymax = max(box1[3], box2[3])
                bwidth = bxmax - bxmin
                bhight = bymax - bymin
                inter = bwidth * bhight
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - inter
                iou = float(inter / max(union, 1e-8))
        else:  # 不相交
            iou = 0
    else:  # 不相交
        iou = 0
    if iou > 1:
        pass
    else:
        return iou


def evaluate_bbox(pre_bboxes, gt_bboxes, process_gt):
    if process_gt:
        gt_bboxes_ = []  # x1, y1, x2, y2
        for gt_bbox in gt_bboxes:
            gt_bboxes_.append([gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]])
    else:
        gt_bboxes_ = gt_bboxes

    iou_bboxes = []
    for i, gt_bbox in enumerate(gt_bboxes_):
        ious = []
        for j, pre_bbox in enumerate(pre_bboxes):
            ious.append(bbox_iou(pre_bbox, gt_bbox))
        if len(ious) > 0:
            iou_bbox = max(ious)
        else:
            iou_bbox = 0
        iou_bboxes.append(iou_bbox)
    miou_bboxes = np.mean(np.array(iou_bboxes))
    return iou_bboxes, miou_bboxes


if __name__ == '__main__':
    file_name = '../params/0-ElBa_selected_annotation.json'
    imgs_info, annotations = get_ElBa_selected_info(file_name)