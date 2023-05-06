# -*- coding:utf-8 -*-
import cv2
from sklearn.cluster import MeanShift,estimate_bandwidth
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def is_similar(pixel_a, cluster_colours):
    # return abs(luminance(pixel_a) - luminance(pixel_b)) < threshold
    # return math.sqrt(r * r + g * g + b * b) < threshold
    color ={}
    color_replaced =[]
    for i in range(len(cluster_colours)):
        r = abs(pixel_a[0] - cluster_colours[i][0])
        g = abs(pixel_a[1] - cluster_colours[i][1])
        b = abs(pixel_a[2] - cluster_colours[i][2])
        c = r+b+g
        color[i] = c
    cc = min(color, key=color.get)
    color_replaced = np.array(cluster_colours[cc])
    return color_replaced


def color_dis(c1, c2):
    r = abs(c1[0] - c2[0])
    g = abs(c1[1] - c2[1])
    b = abs(c1[2] - c2[2])
    v = r + b + g
    return v

def meanshift_seg(files, img_original, savepath):
    files = files[:-4]
    # h = img_original.shape[0]
    # w = img_original.shape[1]
    # h_ = h // 8
    # w_ = w // 8
    # temp_im = cv2.resize(img_original, (h_, w_), interpolation=cv2.INTER_AREA)

    # temp_in = np.array(temp_im, dtype=np.float64)
    # temp_in = np.array(temp_in, dtype=np.float64)
    # colors, counts = np.unique(temp_im.reshape(-1, 3), axis=0, return_counts=1)
    # bandwidth = estimate_bandwidth(colors, quantile=0.3)
    # print(bandwidth)
    colors, counts = np.unique(img_original.reshape(-1, 3), axis=0, return_counts=1)
    bandwidth = estimate_bandwidth(colors, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(colors)
    labels = ms.labels_
    cluster_colors = np.rint(ms.cluster_centers_)
    # print(cluster_colors)

    if len(cluster_colors) > 1:
        # # # replace color
        img2 = np.array(img_original)
        for y in range(img_original.shape[0]):
            for x in range(img_original.shape[1]):
                cc = is_similar(img2[y, x], cluster_colors)  # cc {str}
                img2[y, x] = cc

        # plt.imshow(img2)
        # plt.axis('off')
        # plt.show()
        savename = savepath + files + 'meanshift_.png'
        cv2.imwrite(savename, img2)

        masks = []
        for color in cluster_colors:
            img_ = np.zeros((img_original.shape[0], img_original.shape[1]))
            for y in range(img_original.shape[0]):
                for x in range(img_original.shape[1]):
                    vc = color_dis(img2[y, x], color)  # cc {str}
                    if vc == 0:
                        img_[y, x] = 255
            masks.append(img_.astype(np.uint8))
        return masks
