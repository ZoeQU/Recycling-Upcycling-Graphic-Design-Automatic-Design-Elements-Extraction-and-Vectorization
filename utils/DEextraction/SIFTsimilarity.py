# -*- coding:utf-8 -*-
import cv2
import itertools
import shutil
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import setting


def getMatchNum(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)


def SIFT(img1, img2, processPath, file, k, visualize):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    sift = cv2.xfeatures2d.SIFT_create()
    '''创建FLANN匹配对象'''
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # 索引里的树应该被递归遍历的次数
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    kp1, des1 = sift.detectAndCompute(img1, None)  # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)

    if len(kp1) > 1 and len(kp2) > 1:
        matches = flann.knnMatch(des1, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum, matchesMask) = getMatchNum(matches, 0.9)  # 通过比率条件，计算出匹配程度
        sim = matchNum / len(matches)  # matchNum * 100 / len(matches) 值在[0,1]之间

        if visualize=='True':
            out_put = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
            cv2.imwrite(processPath + file[:-4] + '_' + str(k) + 'sift.png', out_put)
        else:
            pass

    else:
        sim = 0
    return sim