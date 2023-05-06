# -*- coding:utf-8 -*-
import cv2
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw


def anomaly_cut(image, box, cc, k, bk_color, SavePath, elementspath, file):
    # # 构建一个bk_color的image2
    rows = image.shape[0]
    cols = image.shape[1]
    channels = image.shape[2]
    image2 = np.zeros(image.shape, dtype=np.uint8)
    image2[:, :, 0] = np.ones([rows, cols]) * bk_color[2]  # B
    image2[:, :, 1] = np.ones([rows, cols]) * bk_color[1]  # G
    image2[:, :, 2] = np.ones([rows, cols]) * bk_color[0]  # R
    # #不规则多变形裁切
    mask_cut = np.zeros(image.shape, dtype=np.uint8)   # 黑mask
    # #输入点的坐标
    roi_corners = box
    #创建mask层
    cv2.fillPoly(mask_cut, roi_corners, (255, 255, 255))  # mask_cut: 黑mask, 白element
    aa = mask_cut[:, :, 0]
    aa = np.where(aa == 255, 0, 1)
    bb = image2[:, :, 0] * aa
    gg = image2[:, :, 1] * aa
    rr = image2[:, :, 2] * aa
    mask_cut2 = cv2.merge([bb, gg, rr])  # mask_cut: bk_color mask, 黑element
    #为每个像素进行与操作，除mask区域外，全为0
    masked_image = cv2.bitwise_and(image, mask_cut)
    # m2 = cv2.bitwise_and(image2, mask_cut)
    b, g, r = cv2.split(masked_image)
    masked_image2 = cv2.merge([bb + b, gg + g, rr + r])

    # plt.subplot(141)
    # plt.imshow(mask_cut)
    # plt.subplot(142)
    # plt.imshow(mask_cut2)
    # plt.subplot(143)
    # plt.imshow(masked_image)
    # plt.subplot(144)
    # plt.imshow(masked_image2)
    # plt.show()
    # plt.close()

    delements = masked_image2[cc[1]: cc[3], cc[0]: cc[2]]
    cv2.imwrite(elementspath + file[:-4] + '_' + str(k) + '_cc_MBR.png', delements)
    cv2.imwrite(SavePath + file[:-4] + '_' + str(k) + '_cc_MBR.png', masked_image2)

    return masked_image2, delements


def MBR(box):
    """input 格式 ndarray shape (1, n, 2)
    output 格式 [x1,y1,x2,y2]"""
    box_ = box[0].tolist()
    aa = sorted(box_, key=lambda x: x[0])
    minx = aa[0][0]
    maxx = aa[-1][0]
    bb = sorted(box_, key=lambda x: x[1])
    miny = bb[0][1]
    maxy = bb[-1][1]
    return [minx, miny, maxx, maxy]


def irregularCutting(image, binary, bk_color, SavePath, elementsPath, processPath, file, visulization):
    """image: 原图"""
    # 形态学膨胀
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 构建蒙版图片，为了后续可视化
    bottom = np.zeros(image.shape, np.uint8)
    bottom.fill(255)
    top = image.copy()
    overlapping = cv2.addWeighted(bottom, 0.7, top, 0.3, 0)

    de_object_bboxes = []
    de_segments = []
    cuttingImgs = []
    for k in range(len(contours[1])):
        try:
            cont = contours[1][k]
            if cont.shape[0] > 3:
                # 取轮廓长度的1%为epsilon
                epsilon = 0.002 * cv2.arcLength(cont, True)
                # 预测多边形
                box = cv2.approxPolyDP(cont, epsilon, True)
                # 算中心点
                M = cv2.moments(cont)  # 计算第一条轮廓的各阶矩,字典形式
                centerx = int(M["m10"] / (M["m00"] + 0.01))
                centery = int(M["m01"] / (M["m00"] + 0.01))
                new_box = []  # 放大1.1倍
                for i in box:
                    vx = 1.1 * (centerx - i[0][0])
                    vy = 1.1 * (centery - i[0][1])
                    nx = int(math.ceil(centerx - vx))
                    if nx < 0 or nx > image.shape[1]:
                        nx = i[0][0]
                    ny = int(math.ceil(centery - vy))
                    if ny < 0 or ny > image.shape[0]:
                        ny = i[0][1]
                    new_box.append([[nx, ny]])

                new_box = np.array(new_box)

                new_box_ = np.reshape(new_box, (1, -1, 2))
                image4cut = image.copy()

                # #求（不规则）多边形的面积
                mask_area = np.zeros(image.shape[:2], dtype="uint8")
                polygon_mask = cv2.fillPoly(mask_area, new_box_, 255)
                area = np.sum(np.greater(polygon_mask, 0))  # after scale 1.1 times
                area_ori = cv2.contourArea(cont)  # ori_area
                if area_ori > 300 and area_ori < 0.5 * image.shape[0] * image.shape[1]:  # thre=100
                    cc = MBR(new_box_)  # find bbox
                    de_segments.append(new_box)  # 存储数据
                    de_object_bboxes.append(cc)  # 存储数据
                    if cc[3] - cc[1] > 0.067 * image.shape[1] and cc[2] - cc[0] > 0.067 * image.shape[0]:
                        masked_image, delements = anomaly_cut(image4cut, new_box_, cc, k, bk_color, SavePath, elementsPath, file)
                        cuttingImg = [area, masked_image, delements]
                        cuttingImgs.append(cuttingImg)

                        if visulization:# 可视化
                            pts_oribox = box.reshape((-1, 1, 2))
                            img = cv2.polylines(overlapping, pts=[pts_oribox], isClosed=True, color=(31, 23, 176), thickness=1)  # green
                            img = cv2.polylines(img, pts=[new_box_], isClosed=True, color=(103, 54, 16), thickness=2)  # blue
                            cv2.imwrite(processPath + file[:-4] + '_' + str(k) + '_temp_ic.png', img)
        except Exception as e:
            de_segments = None
            de_object_bboxes = None
            cuttingImgs = None
            continue

    if len(cuttingImgs) > (image.shape[0] / 30 * image.shape[1] / 30) or cuttingImgs == None:
        return None, None, None
    else:
        return cuttingImgs, de_object_bboxes, de_segments


