# -*- coding:utf-8 -*-
import setting
import numpy as np
import os
import cv2
import imutils
import pandas as pd
import json
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from utils.DEextraction.UnsupervisedSegmentation import unsupervisedSeg
from utils.DEextraction.irregular_cutting import irregularCutting as ic
from seg_methods.meanshift_seg import meanshift_seg
from seg_methods.kmeans_segmentation import kmeans_seg

def cc_external_expansion(cc, img2):
    """
    为防止detected & extracted 的外接矩形小于图像中的element，对其进行向外扩展10pixel的操作，此操作为了避免扩展后尺寸超过图像边缘
    :param cc: 外接矩形坐标
    :param img2: 原图，三通道图片
    :return: 扩展10pixel后外接矩形坐标
    """
    if (cc[1] - 10) < 0:
        y1 = 0
    else:
        y1 = cc[1] - 10

    if (cc[0] - 10) < 0:
        x1 = 0
    else:
        x1 = cc[0] - 10

    if (cc[1] + cc[3] + 10) > img2.shape[0]:
        y2 = img2.shape[0]
    else:
        y2 = cc[1] + cc[3] + 10

    if (cc[0] + cc[2] + 10) > img2.shape[1]:
        x2 = img2.shape[1]
    else:
        x2 = cc[0] + cc[2] + 10
    return [x1, y1, x2, y2]


def find_cut_coord4boundary(im, img, file, SavePath, processPath):
    """
    找boundary img中各个外界矩形 返回最大矩形
    :param im: 是background mask，单通道图片，其中bk pixel value=255，element pixel value= 0
    :param img: 原图，三通道图片
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param SavePath: 存放地址
    :return: 最大矩形
    """
    im2 = cv2.merge([im, im, im])
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    cut_coord = []
    cnts = contours[0] if imutils.is_cv2() else contours[1]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)          # 外接矩形框，没有方向角 左上角（x,y）右下角(x+w, y+h）
        if im.shape[1] - w > 10 and im.shape[0] - h > 10:
            # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print([x, y, w, h])
            cut_coord.append([x, y, w, h, w * h])  # 左上角，右下角
    # cv2.imwrite(setting.SavePath + file[:-4] + "contours_canny" + ".png", im2)  # 中间结果，就是看看

    boundary_cc = max(cut_coord, key=lambda item: item[4])

    # cc = cc_external_expansion(boundary_cc, img)
    # cv2.rectangle(im2, (cc[0], cc[1]), (cc[2], cc[3]), (0, 255, 0), 2)
    # cv2.imwrite(processPath + file[:-4] + "contours_canny" + ".png", im2)  # 中间结果，就是看看
    # boundary_MBR = img[cc[1]: cc[3], cc[0]: cc[2]]
    # cv2.imwrite(SavePath + file[:-4] + '_' + 'boundary_MBR' + '.png', boundary_MBR)  # extracted elements 保存到指定文件夹内

    cc = img[boundary_cc[1]: boundary_cc[1] + boundary_cc[3],
             boundary_cc[0]: boundary_cc[0] + boundary_cc[2]]
    cv2.imwrite(SavePath + file[:-4] + '_' + 'boundary_MBR' + '.png', cc)


def MBRCutting(bk_mask, file, img, processPath, SavePath):
        cut_coord = find_cut_coord(bk_mask, file, img, processPath)
        if len(cut_coord) > 0:
            i = 0
            for cc in cut_coord:
                img2 = img.copy()
                # cc_MBR = img2[cc[1] - 10: cc[1] + cc[3] + 10, cc[0] - 10: cc[0] + cc[2] + 10]
                cc = cc_external_expansion(cc, img2)  # [x1,y1,x2,y2]
                cc_MBR = img2[cc[1]: cc[3], cc[0]: cc[2]]
                cv2.imwrite(SavePath + file[:-4] + '_' + str(i) + '_cc_MBR.png', cc_MBR)
                i += 1


def find_cut_coord(im0, file, img, processPath):
    """    the function 用于 detect & extract img中的elements.
    :param im0:  是background mask，单通道图片，其中bk pixel value=255，element pixel value= 0
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param img: 原图，三通道图片
    :param processPath: 存放地址
    :return: 每个矩形区域(作为一个个独立的element)    """
    im = 255 - im0  # invert 后 bk pixel value=0，element pixel value= 255
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    cut_coord = []
    cnts = contours[0] if imutils.is_cv2() else contours[1]

    img0 = img.copy()

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形框，没有方向角 左上角（x,y）右下角(x+w, y+h）
        cut_coord.append([x, y, w, h, w * h])  # 左上角，右下角
        # print([x, y, w, h])

    cut_coord = sorted(cut_coord, key=(lambda x: x[0]))
    for i in range(len(cut_coord)):  # [x, y, w, h]
        for j in range(i + 1, len(cut_coord)):
            cut_coord[i], cut_coord[j] = ifInclude(cut_coord[i], cut_coord[j])

    cut_coord2 = []
    for k in cut_coord:
        x2 = k[0]
        y2 = k[1]
        w2 = k[2]
        h2 = k[3]
        if x2 != 0 and y2 != 0 and x2 + w2 < im.shape[1] and y2 + h2 < im.shape[0]:
            cut_coord2.append(k)
            cv2.rectangle(img0, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    cv2.imwrite(processPath + file[:-4] + "_contours_bk.png", img0)
    return cut_coord2


def find_bk_by_cut_coord(imgs):
    """    判断这个im(mask)是否和img本身尺寸相仿,若相仿则返回im本身
    :param imgs: masks
    :return: bk_mask    """
    bk_candidates = []
    num_bk_pixel = []
    for im in imgs:
        flatten_mask = im.flatten()
        num_bk_pixel.append(np.count_nonzero(flatten_mask))

        contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
        cnts = contours[0] if imutils.is_cv2() else contours[1]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形框，没有方向角 左上角（x,y）右下角(x+w, y+h）
            if im.shape[1] - w < 10 and im.shape[0] - h < 10:
                bk_candidates.append(im)
                break

    if len(bk_candidates) == 0:
        bk_mask = imgs[np.argmax(np.array(num_bk_pixel))]
    elif len(bk_candidates) > 1:
        axis_2 = np.count_nonzero(np.array(bk_candidates), axis=2)
        s2 = np.sum(axis_2, axis=1)
        bk_mask = bk_candidates[np.argmax(s2)]
    else:
        bk_mask = bk_candidates[0]
    return bk_mask


def isIntersection(a, b):
    """    element_a与element_b是否相交、包含, 未用
    :param a: element_a的坐标
    :param b: element_b的坐标
    :return: 合并后的new坐标    """
    if a[0] < b[0]:
        if b[0] < (a[0] + a[2]) and b[1] < (a[1] + a[3]) and b[1] > a[1]:
            d = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            return d

    if a[0] > b[0]:
        if a[0] < (b[0] + b[2]) and a[1] < (b[1] + b[3]) and a[1] > b[1]:
            d = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            return d

    if a[0] == b[0]:
        d = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        return d


def ifInclude(a, b):
    """    element_a与element_b是否包含
    :param a: element_a的坐标
    :param b: element_b的坐标
    :return: 合并后的new坐标    """
    if a[0] <= b[0]:
        if b[0] <= (a[0] + a[2]) and b[1] <= (a[1] + a[3]) and (b[0] + b[2]) < (a[0] + a[2]) and (b[1] + b[3]) < (a[1] + a[3]):
            b = [0, 0, 0, 0, 0]
            return a, b
        else:
            return a, b

    else:
        if a[0] < (b[0] + b[2]) and a[1] < (b[1] + b[3]) and (a[0] + a[2]) < (b[0] + b[2]) and (a[1] + a[3]) < (b[1] + b[3]):
            a = [0, 0, 0, 0, 0]
            return a, b
        else:
            return a, b


def findBK_mask(img, dim2, file, type, processPath, maxIter, visualization):
    """通过3次unsupervised segmentation得到稳定的bk_mask
    :param img: 原图，三通道图片
    :param dim2: 图像尺寸 (1, 3, width, height),unsupervised segmentation 需要
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param type: 初始分割的方法 [SLIC 和 felzenszwalb]
    :param maxIter: 迭代多少次,初始50次
    :param visualization: 可视化，存中间图片
    :return: bk_mask"""


    # 1. unsupervised
    im_target = unsupervisedSeg(input=img, maxIter=maxIter, dim=dim2, name=file[:-4], type=type,
                                color_consistency=False, processPath=processPath, visualization=visualization)

    masks = []
    colors = np.unique(im_target)
    num_color = len(colors)
    for k in range(num_color):
        color = colors[k]
        im_target_ = im_target.copy()
        im_target_ = np.where(im_target_ == color, 255, 0)  # white|255:content black|0:background
        im_target2_ = np.where(im_target == color, 1, 0)  # save color mask [0,1]

        im_target_ = im_target_.reshape((img.shape[0], img.shape[1])).astype(np.uint8)
        masks.append(im_target_)

        if visualization: #save rgb bk mask
            im_target2_ = im_target2_.reshape((img.shape[0], img.shape[1])).astype(np.uint8)
            b, g, r = cv2.split(img)
            im_target2_ = cv2.merge([im_target2_ * b, im_target2_ * g, im_target2_ * r])
            cv2.imwrite(processPath + file[:-4] + '_out_' + str(k) + '.png', im_target2_)  # save each color mask

    """
    # 2. meanshift-test
    masks = meanshift_seg(files=file, img_original=img, savepath=processPath)
    
    # 3. kmeans, k=2
    masks = kmeans_seg(files=file, img_original=img, savepath=processPath)
    """

    # which one is the background mask"""
    bk_mask = find_bk_by_cut_coord(masks)
    return bk_mask


def visualPieChart(sim, processPath, method):
    y = np.array(sim)
    bins = pd.IntervalIndex.from_tuples([(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25),
                                         (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5),
                                         (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75),
                                         (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)])
    y_cut = pd.cut(y, bins)
    y_score = pd.value_counts(y_cut)
    y_score2 = dict(y_score)
    labels = []
    values = []
    for key, value in y_score2.items():
        if value > 0:
            labels.append(str(key))
            values.append(value)

    plt.pie(np.array(values), labels=labels, autopct='%.2f%%')
    plt.title("Similarity Pie Chart Distribution")
    plt.savefig(processPath + str(method) + " similarity_pie_chart.svg")
    # plt.show()
    plt.close('all')


def visualHist(sim, processPath, method):
    n = len(sim)
    plt.hist(sim, bins=n, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("times")
    plt.ylabel("sim value")
    plt.title("similarity values")
    plt.savefig(processPath + str(method) + " similarity values.svg")
    # plt.show()
    plt.close('all')


def img_resize(img, scale_percent):
    """resize image"""
    scale_percent = scale_percent  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

