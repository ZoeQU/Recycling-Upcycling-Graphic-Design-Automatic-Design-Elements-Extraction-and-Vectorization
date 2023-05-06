# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import time
import csv
import pandas as pd


processPath = 'doc/de-sift/'
if not os.path.exists(processPath):
    os.mkdir(processPath)


res = []
k = 0
for files in os.listdir('doc/design_elements'):
    print("image name: %s" % files)
    start_time = time.time()
    img = cv2.imread('doc/design_elements/' + files)

    sift = cv2.xfeatures2d.SIFT_create()
    '''创建FLANN匹配对象'''
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # 索引里的树应该被递归遍历的次数
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    kp, des = sift.detectAndCompute(img, None)

    bottom = np.zeros(img.shape, np.uint8)
    bottom.fill(255)
    top = img.copy()
    overlapping = cv2.addWeighted(bottom, 0.5, top, 0.5, 0)
    # 在关键点的位置上绘制小圆圈
    overlapping = cv2.drawKeypoints(overlapping, kp, img)
    # 在关键点的位置上绘制一个大小为keypoint的小圆圈
    # img = cv2.drawKeypoints(gray , kp , img , flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(processPath + files[:-4] + str(k) + '_SIFTpoints' + '.png', overlapping)
    timecost = round(time.time() - start_time, 2)
    res.append([files, timecost])
    k += 1



res_name = processPath + 'input_cropedpattern_results_SIFT.csv'
title = ['de_name, time']
with open(res_name, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    writer.writerows(res)

print('finish')