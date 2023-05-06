# -*- coding:utf-8 -*-
import json
import os
import numpy as np
from utils.data_process.parse_ElBa_jason import GetJsonInfo, get_img_info
from utils.data_process.calculate_IoU import eval_segmentation, eval_bkmask, evaluate_bbox
import gc
import pandas as pd
from tqdm import tqdm, trange
import time

from utils.data_process.calculate_IoU import eval_segmentation, eval_bkmask, evaluate_bbox
import setting


def get_annotations(fileName):
    fileName = fileName[:-3] + 'json'
    annotationsPath = 'utils/params/2-input-annotations/'   # 126 images

    with open(os.path.join(annotationsPath, fileName), 'r', encoding='utf-8') as f:
        temp = json.load(f)
    data = temp["shapes"]
    imageHeight = temp["imageHeight"]
    imageWidth = temp["imageWidth"]
    labels = set()
    bboxes = []
    for d in data:
        label = d["label"]; labels.add(label)
        points = np.array(d["points"]).reshape(1, -1)
        bboxes.append(points[0])
    return labels, bboxes


fileName = 'doc/output/seg_results_SLIC10000_json.json'
with open(fileName, 'r', encoding='utf-8') as f:
    temp = json.load(f)
results = temp["results"]

save_results = []
for n in trange(len(results), desc='Calculating'):
    time.sleep(0.1)
    result = results[n]

    imageName = result["image_name"]
    segmentationTime = result["seg_time"]
    extractionTime = result["design_elements_extraction_time"]

    pre_bboxes = []
    bbox = result["bbox"]
    for bb in bbox:
        pre_bboxes.append(bb["coords"])

    # evaluate
    labels, gt_bboxes = get_annotations(imageName)
    iou_bboxes, bboxes_miou = evaluate_bbox(pre_bboxes, gt_bboxes, False)
    temp = [imageName, segmentationTime, extractionTime, bboxes_miou]

    # accuracy, precision, recall, ...
    iou_bboxes = np.array(iou_bboxes)
    thres = np.linspace(0.1, 1, 9, endpoint=False)
    for thre in thres:
        tp = np.sum(iou_bboxes >= thre)
        tn = np.sum(iou_bboxes < thre)
        fp = fn = 0
        accuray = tp / (tp + tn)
        temp.append(accuray)


    save_results.append(temp)

# 按行写入
titles = ['image_name', 'seg_time', 'de_extraction_time', 'bboxes_miou',
          'acc0.1', 'acc0.2', 'acc0.3', 'acc0.4', 'acc0.5', 'acc0.6',
          'acc0.7', 'acc0.8', 'acc0.9']

df = pd.DataFrame(save_results, columns=titles, dtype=float)
df.to_excel(os.path.join(setting.SavePath, 'our_dataset_SLIC10000_bbox_results.xlsx'))



