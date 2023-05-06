# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def cal_rectangle(im):
    """
    find the bounding rectangle + remove small regions first
    :param im:
    :return:
    """
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.erode(im, element, iterations=1)
    mask = cv2.dilate(im, element, iterations=1)
    mask = cv2.erode(mask, element)
    coordinate = np.where(mask == 0)
    y0 = min(coordinate[0])
    h0 = max(coordinate[0])
    x0 = min(coordinate[1])
    w0 = max(coordinate[1])
    # merged_binary_rectangle = im[y0:y0 + h0, x0:x0 + w0]
    CC_rectangle = [x0, w0, y0, h0]
    return CC_rectangle


def cal_points_distance(p1, p2):
    """p1和p2之间的欧式距离"""
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


def cal_centroid(img_binary):
    """
    计算一个闭合区域的中心点（centroid_point）
    :param img_binary: 必须是一张二值图片（白色为内容，黑色为背景）
    :return:
    """
    m = cv2.moments(img_binary)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    centroid = [x, y]
    return centroid


def cal_area(img):
    """
    计算背景颜色所占的面积比 (pixel_value = 0 / [0, 0, 0])
    :param img: 图片
    :return:
    """
    # # 3d img method 1 ##
    # c, amount_m = np.unique(merged.reshape(-1, 3), axis=0, return_counts=1)
    # cc_amount = amount_m[c.tolist().index(cc)]
    # arearate = cc_amount/(im.shape[0]*im.shape[1])

    # # img method 2 ##
    if len(img.shape) == 2:
        area = np.sum(img == 0)
        arearate = area / (img.shape[0]*img.shape[1])
    else:
        r, g, b = cv2.split(img)
        im = r + g + b
        area = np.sum(im == 0)
        arearate = 1.0 - area / (img.shape[0]*img.shape[1])
    return arearate


def cal_pixel_distribution(img, cc):
    """
    计算目标pixel value在一张图片中的分布(将图片分成16*16的网格)情况
    :param img: 图片
    :param cc: 目标pixel value
    :return: 在网格中的占比
    """
    if len(img.shape) == 3:
        # # if the image is 3D shape, we calculate pixel value == cc
        width, height = img.shape[1], img.shape[0]
        item_width = int(width / 16)
        item_height = int(width / 16)
        boxnum = 0
        # (y0, y1, x0, x1)
        for i in range(0, 16):  # 两重循环，生成16x16张图片基于原图的位置
            for j in range(0, 16):
                # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
                box = img[j * item_height:(j + 1) * item_height, i * item_width:(i + 1) * item_width]
                amount = np.where(box == cc)[0].shape[0] / 3
                if amount > 0:
                    boxnum += 1
    else:
        # # if the image is 2D shape, we calculate pixel value == 0
        width, height = img.shape[1], img.shape[0]
        item_width = int(width / 16)
        item_height = int(width / 16)
        boxnum = 0
        # (y0, y1, x0, x1)
        for i in range(0, 16):  # 两重循环，生成16x16张图片基于原图的位置
            for j in range(0, 16):
                # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
                box = img[j * item_height:(j + 1) * item_height, i * item_width:(i + 1) * item_width]
                amount = np.where(box == cc)[0].shape[0] / 3
                if amount > 0:
                    boxnum += 1
    return boxnum / 256


def ColourDistance(rgb_1, rgb_2):
    """
    比较2个rgb颜色值(转换成hsv)后之间的距离
    :param rgb_1: rgb1颜色值
    :param rgb_2: rgb2颜色值
    :return: 距离值
    """
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


def RGB_to_Hex(rgb):
    """将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示"""
    # RGB = rgb.split(',')            # 将RGB格式划分开来 str format
    color = '#'
    for i in rgb:
        if np.isnan(i):
            i = 255
        else:
            num = int(i)
            color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


def Hex2Rgb(value):
    """将16进制Hex 转化为 [R,G,B]"""
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def color_blocks(cluster_rgb, savename):
    """
    show color blocks, max 10 colors
    :param cluster_rgb: colors,max 10 colors
    :param savename: savename, str
    :return:
    """
    color_fig = plt.figure()
    box = color_fig.add_subplot(111, aspect='equal')
    for i in range(len(cluster_rgb)):
        face_color = tuple(np.array(cluster_rgb[i]) / 255)
        loc_x = i * 0.1
        Loc = tuple([loc_x, 0])
        tmp_box = plt.Rectangle(Loc, 0.1, 0.8, facecolor=face_color, edgecolor='r', fill=True)
        box.add_patch(tmp_box)
        plt.text(loc_x, 0.95, "["+str(cluster_rgb[i][0]))
        plt.text(loc_x, 0.9, str(cluster_rgb[i][1]))
        plt.text(loc_x, 0.85, str(cluster_rgb[i][2])+"]")
    plt.axis('off')
    color_fig.savefig(savename)
    # plt.show()
    plt.close(color_fig)


def color_his(img, files, processPath, visualization):
    """
    通过统计颜色直方图得到主要颜色的数目
    :param img: 原图，三通道
    :param files: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param processPath: 存放地址
    :param visualization: 可视化
    :return:
    """
    img = img.reshape((-1, 1))

    # # color histogram v1
    # l = 12
    # fig, ax = plt.subplots(figsize=(10, 7))
    # p = ax.hist(img, bins=l, rwidth=0.45, weights=[1. / len(img)] * len(img))
    # ax.set_title("Color Histogram")
    # # plt.axhline(y=0.02, xmin=0.0, c='skyblue', ls='--', linewidth=1)
    # plt.savefig(processPath + files[:-4] + '_' + "gray_color_dis_hist.eps")
    # plt.show()
    # plt.close()

    # num = []
    # mea = np.mean(hist_)
    # std = np.std(hist_)
    # for item in B:
    #     if std > 0.08:
    #         if item > 0.03:  #ori_2: 0.03
    #             num.append(item)
    #     else:
    #         if item > mea:
    #             num.append(item)
    #
    # if len(num) > 8:
    #     return 4
    # else:
    #     return len(num)

    # # color histogram v2
    hist = cv2.calcHist([img], [0], None, [26], [0.0, 256.0])
    minVal_b, maxVal_b, minLoc_b, maxLoc_b = cv2.minMaxLoc(hist)
    A = np.sum(hist)
    hist_ = hist / A
    B = hist_.flatten().tolist()

    if visualization:
        plt.bar(range(len(B)), B)
        plt.title('Grayscale Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of Pixels')
        plt.savefig(processPath + files[:-4] + '_' + "gray_color_dis_hist.eps")
        # plt.show()
        plt.close('all')

    num = []
    for item in B:
        if item > 0.02:
            num.append(item)

    if len(num) > 1:
        return len(num)
    else:
        return 2
