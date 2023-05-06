# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np

def adaptive_threshod_segmentation(files, img_original, type_, savepath):
    """
    自适应图片分割法，看看和unsupervised segmentation那个好用
    :param files: 图片名
    :param img_original: 原图
    :param savepath:
    :return:
    """

    files = files[:-4]
    if type_ == 'global':
        #1. 全局阈值分割
        retval, img_global = cv2.threshold(img_original, 130, 255, cv2.THRESH_BINARY)
        labels = np.unique(img_global)
        savename = savepath + files + '_thre_global.png'
        cv2.imwrite(savename, img_global)

        return labels

    elif type_ == 'adaptive':
        #2. 自适应阈值分割
        img_ada_mean = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
        labels_ada_mean = np.unique(img_ada_mean)
        img_ada_gaussian = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        labels_ada_gaussian = np.unique(img_ada_gaussian)

        savename_mean = savepath + files + '_thre_mean_ada.png'
        cv2.imwrite(savename_mean, img_ada_mean)

        savename_gaussian = savepath + files + '_thre_gaussi_ada.png'
        cv2.imwrite(savename_gaussian, img_ada_gaussian)

        return labels_ada_mean, labels_ada_gaussian

        # #显示图片
    # imgs = [img_original, img_global, img_ada_mean, img_ada_gaussian]
    # titles = ['Original Image', 'Global Thresholding(130)', 'Adaptive Mean', 'Adaptive Guassian']
    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     plt.imshow(imgs[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
