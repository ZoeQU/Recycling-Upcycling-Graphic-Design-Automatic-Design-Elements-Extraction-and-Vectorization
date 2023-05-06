# -*- coding:utf-8 -*-
import setting
import numpy as np
import itertools
import os
import time
import cv2
# print(cv2.__version__)
import imutils
import random
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from utils.DEextraction.irregular_cutting import irregularCutting as ic
from utils.DEextraction.deefuncitions import (find_cut_coord4boundary, img_resize,
                                              findBK_mask, MBRCutting, visualHist, visualPieChart)
from utils.DEextraction.perceptualhashing import PerceptualHashingSimilarity
from utils.DEextraction.SIFTsimilarity import SIFT
from utils.DEextraction.CNNFeaturesimilarity import CNNfeaturesSim

import warnings
warnings.filterwarnings("ignore")


def mkfolder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def graphic_pattern_background_extraction(file, start_time, paths, t, initialtype, visualization, maxIter=50):
    SavePath, processPath, elementsPath, svgPath, keepPath = paths
    img = cv2.imread('./' + setting.ImgPath + file)  # 读取图片

    """resize image"""
    scale = 100  # todo(): if need change the scale
    if scale != 100:
        img = img_resize(img, scale)

    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    dim = (1, 3, img.shape[1], img.shape[0])

    if t > 1:
        temp_bk = []
        bk_mask = np.zeros((img.shape[0], img.shape[1]))
        for n in range(t):
            bkmask = findBK_mask(img, dim, file, initialtype, processPath, maxIter, visualization)
            bk_mask += bkmask / 255
            temp_bk.append(bkmask)  # each sub_bk mask
        bk_mask = np.where(bk_mask.flatten() > (t - 1), 255, 0).reshape((img.shape[0], img.shape[1])).astype(np.uint8)
        temp_bk.append(bk_mask)  # final bk mask
    else:
        temp_bk = []
        bk_mask = findBK_mask(img, dim, file, initialtype, processPath, maxIter, visualization)
        temp_bk.append(bk_mask); temp_bk.append(bk_mask)  # 就写两次 不能删

    if visualization:   # bk masks visualization
        titles = ['{} time'.format(i+1) for i in range(t)]
        titles.append('final bk mask')
        for i in range(t + 1):
            plt.subplot(1, t + 1, i + 1)
            plt.imshow(temp_bk[i], 'gray')
            plt.title(titles[i])
            # plt.axis('off')
        plt.savefig(processPath + file[:-5] + "bk_masks" + ".png")
        # plt.show()
        plt.close('all')

    seg_time = round(time.time() - start_time, 1)
    return bk_mask, seg_time


def ForegroundElementsExtraction(file, start_time, t, canny, MBR, initialtype, visualization, maxIter=50):
    """根据unsupervised segmentation的结果找到所有element（不管重复与否，不论尺寸大小）,返回一个地址（文件夹）
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param start_time: 算法初始时间
    :param t: 分割迭代次数
    :param canny: 是否利用canny edge结果辅助element extraction，默认False
    :param initialtype: 初始分割的方法 [SLIC 和 felzenszwalb]
    :param visualization: 是否可视化每次unsupervised segmentation的结果，默认false
    :param maxIter: 迭代多少次,初始50次
    :return: 存放extracted elements文件夹地址 & bk_color (rgb)"""

    SavePath = setting.SavePath + str(file[:-4]) + '/'
    processPath = setting.SavePath + file[:-4] + '/process/'
    elementsPath = setting.SavePath + file[:-4] + '/elements/'
    svgPath = setting.SavePath + file[:-4] + '/svg/'
    keepPath = setting.SavePath + file[:-4] + '/' + 'keep/'

    paths = [SavePath, processPath, elementsPath, svgPath, keepPath]
    for folder in paths:
        mkfolder(folder)

    img = cv2.imread('./' + setting.ImgPath + file)  # 读取图片

    if canny:
        """Canny edge作为辅助信息 element extraction"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        boundary_ = cv2.Canny(image=img_blur, threshold1=10, threshold2=50)  # Canny Edge Detection
        find_cut_coord4boundary(boundary_, img, file, SavePath, processPath)  # name: contours_canny
    else:
        pass

    """resize image"""
    scale = 100  # todo(): if need change the scale
    if scale != 100:
        img = img_resize(img, scale)

    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    dim = (1, 3, img.shape[1], img.shape[0])

    bk_mask, seg_time = graphic_pattern_background_extraction(file, start_time, paths, t, initialtype, visualization, maxIter=50)

    """bk_color RGB value extraction by RGB mean value"""
    B, G, R = cv2.split(img)
    bk_01 = np.where(bk_mask == 255, 1, 0)  # 将bk_mask【0，1】化
    B_ = int(B[np.nonzero(bk_01 * B)].mean())
    G_ = int(G[np.nonzero(bk_01 * G)].mean())
    R_ = int(R[np.nonzero(bk_01 * R)].mean())
    bk_color = [R_, G_, B_]  # rbg value
    rr = bk_01 * R_
    gg = bk_01 * G_
    bb = bk_01 * B_
    im_bk_rgb = cv2.merge([bb, gg, rr])

    if visualization:  # bk_color rgb mask
        cv2.imwrite(processPath + file[:-5] + 'bk_mask_rgb.png', im_bk_rgb)

    """elements extraction"""
    if MBR:  # bbox cuting
        MBRCutting(bk_mask, file, img, processPath, SavePath)
        print('crop element!')
        return SavePath, bk_color, processPath, seg_time
    else:
        """irregular cutting"""
        cuttingImgs, de_object_bboxes, de_segments = ic(img, bk_mask, bk_color, SavePath, elementsPath, processPath, file, visualization)

        if cuttingImgs == None:
            cuttingImgs = img

        if de_object_bboxes == None:
            h = img.shape[0]
            w = img.shape[1]
            de_object_bboxes = [0, 0, w, h]
            de_segments = [[0, 0], [w, 0], [w, h], [0, h]]

        print('crop element!')
        return bk_color, cuttingImgs, seg_time, paths, bk_mask, de_segments, de_object_bboxes


def RedundantElementsRemoval(file, cuttingImgs, paths, method, visualization):
    '''    比较elements间CNN feature distance，用于剔除冗余elements
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param cuttingImgs: [裁切下来的图片mask的面积，mask本身]
    :param paths: [路径们]
    :param method: 衡量相似度的方法
    :param visualization: 可视化(每对element的相似度的集合：)similar_value 的曲线，默认为False
    :return: 剔除冗余后的elements |repeated elements| Group[list] 和 栅格化后的elements图片地址    '''
    SavePath, processPath, elementsPath, svgPath, keepPath = paths
    Group = cuttingImgs  # just change the name

    if len(Group) == 1:
        name = keepPath + file[:-5] + '_' + str(0) + '.png'
        cv2.imwrite(name, Group[0][2])
        print('keep elements: %s' % len(Group))
        return keepPath, Group

    else:
        if method == "Perceptual Hashing":
            GG = list(itertools.combinations(Group, 2))
            sim = []
            for gg in GG:
                contentMatchRatio = PerceptualHashingSimilarity(gg[0][2].astype('uint8'), gg[1][2].astype('uint8'))
                similarity = np.mean(contentMatchRatio[:-2])
                sim.append(similarity)

            thre = 0.8

            if visualization:
                # visualHist(sim, processPath, "Perceptual Hashing")
                visualPieChart(sim, processPath, "Perceptual Hashing")

            Group2 = Group.copy()
            K = []
            while len(Group2) > 0:
                Group2 = sorted(Group2, key=lambda x: x[0])  # area从小到大排序
                t = Group2[-1]
                K.append(Group2[-1])
                Group2 = Group2[:-1]

                dell = []
                for i in range(len(Group2)):
                    contentMatchRatio = PerceptualHashingSimilarity(Group2[i][2], t[2])
                    similarity = np.mean(contentMatchRatio[:-2])
                    if similarity > thre:
                        dell.append(Group2[i])

                if len(dell) > 0:
                    for k in range(len(dell)):
                        try:
                            Group2.remove(dell[k])
                        except Exception as e:
                            continue
                else:
                    pass

            if len(K) > 0:
                for q in range(len(K)):
                    name = keepPath + file[:-5] + '_' + str(q) + '.png'
                    cv2.imwrite(name, K[q][2])
                print('keep elements: %s' % len(K))
            else:
                name = keepPath + file[:-5] + '_' + str(0) + '.png'
                img = cv2.imread(setting.ImgPath + file)
                cv2.imwrite(name, img)
                print('keep elements: %s' % 1)
            return keepPath, Group

        elif method == "CNN features":
            GG = list(itertools.combinations(Group, 2))
            sim = []
            for gg in GG:
                contentMatchRatio = CNNfeaturesSim(gg[0][1], gg[1][1])
                similarity = np.mean(contentMatchRatio)
                sim.append(similarity)

            sim_max = max(sim)
            thre = 0.85 * sim_max

            if visualization:
                visualHist(sim, processPath, "CNN features")

            p = 0
            j = 1
            while p < len(Group):
                group = sorted(Group, key=lambda x: x[0])  # area从小到大排序
                dell = []
                for i in range(len(group) - j):
                    similarity = CNNfeaturesSim(group[0][1], group[1][1])
                    if similarity > thre:
                        dell.append(group[i])
                for k in dell:
                    try:
                        Group.remove(k)
                    except Exception as e:
                        continue
                j += 1
                p += 1

            for q in range(len(Group)):
                name = keepPath + file[:-5] + '_' + str(q) + '.png'
                cv2.imwrite(name, Group[q][2])

            print('keep elements: %s' % len(Group))
            return keepPath, Group

        else:
            """SIFT"""
            GG = list(itertools.combinations(Group, 2))
            sim = []
            k = 0
            for gg in GG:
                similarity = SIFT(gg[0][1], gg[1][1], processPath, file, k, True)
                k += 1
                sim.append(similarity)

            thre = 0.85

            if visualization:
                visualHist(sim, processPath, "SIFT")

            p = 0
            j = 1
            while p < len(Group):
                group = sorted(Group, key=lambda x: x[0])  # area从小到大排序
                dell = []
                for i in range(len(group) - j):
                    similarity = SIFT(group[0][1], group[1][1], processPath, file, k, False)
                    if similarity > thre:
                        dell.append(group[i])
                for k in dell:
                    try:
                        Group.remove(k)
                    except Exception as e:
                        continue
                j += 1
                p += 1

            for q in range(len(Group)):
                name = keepPath + file[:-5] + '_' + str(q) + '.png'
                cv2.imwrite(name, Group[q][2])

            print('keep elements: %s' % len(Group))

            return Group


