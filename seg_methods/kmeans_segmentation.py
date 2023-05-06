# -*- coding:utf-8 -*-
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def kmeans_seg(files, img_original, savepath):
    files = files[:-4]
    temp_im = np.array(img_original, dtype=np.float64) / 255
    w, h, c = temp_im.shape
    dd = temp_im.reshape(w * h, c)
    Kcolor = KMeans(n_clusters=2)
    Kcolor.fit(dd)
    label_pred = Kcolor.predict(dd)
    cluster_colors = Kcolor.cluster_centers_
    img_clustered = label_pred.reshape((w, h)).astype(np.uint8)
    savename = savepath + files + 'kmeans_.png'
    cv2.imwrite(savename, img_clustered)

    mask_1 = np.where((img_clustered == 0), 255, 0)
    mask_2 = np.where((img_clustered == 1), 255, 0)
    masks = np.array([mask_1, mask_2]).astype(np.uint8)

    # plt.subplot(121)
    # plt.imshow(mask_1)
    # plt.subplot(122)
    # plt.imshow(mask_2)
    # plt.show()
    return masks


if __name__ == '__main__':
    fileName = '1af8c6178ae6b17c7f37b0beff0ee791.jpg'
    img_original = cv2.imread('../doc/2-input/1af8c6178ae6b17c7f37b0beff0ee791.jpg')
    savepath = ''
    masks = kmeans_seg(fileName, img_original, savepath)