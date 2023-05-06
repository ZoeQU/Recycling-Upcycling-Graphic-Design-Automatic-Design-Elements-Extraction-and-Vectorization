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

import setting

# load results
file_name = os.path.join('doc/output/', 'seg_results_SLIC1000_json.json')
with open(file_name, 'r') as load_f:
    data = json.load(load_f)
    results = data['results']

# load GT
parser = GetJsonInfo('utils/params/0-ElBa_selected_annotation.json')
imgs, annotations = parser.read_json()


def evaluate(result):
    image_name = result['image_name']
    seg_time = result['seg_time']
    de_extraction_time = result['design_elements_extraction_time']

    bk_mask = result['bk_mask']
    de_segments = result['segments']
    segments = []
    for i, segs in enumerate(de_segments):
        segments.append(np.array(segs['coords']).reshape(-1, 2))

    de_bboxes = result['bbox']
    bboxes = []
    for j, de_bbox in enumerate(de_bboxes):
        bboxes.append(de_bbox['coords'])
    # load its gt info
    im_info, im_annotation, im_segmentations, im_bboxes = get_img_info(image_name, imgs, annotations)
    # evaluate segmentation
    seg_eval_results = eval_segmentation(pred_segments=segments, gt_segments=im_segmentations)

    seg_result_iou_all_categs = seg_eval_results['iou_all_categs']
    # accuracy, precision, recall, ...
    iou_bboxes = np.array(seg_result_iou_all_categs)
    thres = np.linspace(0.1, 1, 9, endpoint=False)
    Acc = []
    for thre in thres:
        tp = np.sum(iou_bboxes >= thre)
        tn = np.sum(iou_bboxes < thre)
        fp = fn = 0
        accuray = tp / (tp + tn)
        Acc.append(accuray)

    seg_miou = seg_eval_results['mIoU']
    # evaluate bk_mask
    bk_iou = eval_bkmask(pre_bk=bk_mask, img_name=image_name, imgs_segs=im_segmentations)
    # evaluate bbox
    iou_bboxes, bboxes_miou = evaluate_bbox(pre_bboxes=bboxes, gt_bboxes=im_bboxes, process_gt=True)
    return image_name, seg_time, de_extraction_time, seg_miou, bboxes_miou, bk_iou, Acc


save_results = []
for i in trange(len(results), desc='Calculating'):
    result = results[i]
    image_name, seg_time, de_extraction_time, seg_miou, bboxes_miou, bk_iou, Acc = evaluate(result)
    temp = [image_name, seg_time, de_extraction_time, seg_miou, bboxes_miou, bk_iou]
    for acc in Acc:
        temp.append(acc)
    save_results.append(temp)
    time.sleep(1)
    gc.collect()

# 按行写入
titles = ['image_name', 'seg_time', 'de_extraction_time', 'seg_miou', 'bboxes_miou', 'bk_iou',
          'acc0.1', 'acc0.2', 'acc0.3', 'acc0.4', 'acc0.5', 'acc0.6', 'acc0.7', 'acc0.8', 'acc0.9']
df = pd.DataFrame(save_results, columns=titles, dtype=float)
df.to_excel(os.path.join(setting.SavePath, 'elba_seg_slic1000overall_results.xlsx'))

