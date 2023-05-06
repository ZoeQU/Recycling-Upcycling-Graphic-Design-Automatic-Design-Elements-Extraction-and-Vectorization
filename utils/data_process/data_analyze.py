# -*- coding:utf-8 -*-

import numpy as np
import cv2
import os
import csv
import seaborn as sns
import palettable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



def draw_scatter(sizes):
    w = np.array([i[0] for i in sizes])
    h = np.array([i[1] for i in sizes])
    plt.scatter(w, h)

    # ax = plt.gca()
    # x = MultipleLocator(20)
    # y = MultipleLocator(20)
    # ax.xaxis.set_major_locator(x)
    # ax.yaxis.set_major_locator(y)
    # plt.xlim((0, 110))
    # plt.ylim((0, 110))
    # plt.axis([0, 110, 0, 110])

    plt.savefig('dataset_sizes.png')
    plt.show()
    plt.clf()


def heatmap(sizes, sample, n, head):
    if n == '1':
        # 1. for img sizes
        w_edges = range(int(head[1]), int(head[3]), 20)
        h_edges = range(int(head[5]), int(head[7]), 20)
        w = np.array([int(i[0]) for i in sizes])
        h = np.array([int(i[1]) for i in sizes])
        heatmap, xedges, yedges = np.histogram2d(w, h, bins=(w_edges, h_edges))

        if sample:
            # # 1.express edition figure

            # color palette 1
            # top = cm.get_cmap('Oranges_r', 128)
            # bottom = cm.get_cmap('Blues', 128)
            # newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
            # newcmp = ListedColormap(newcolors, name='OrangeBlue')

            # # color palette 2
            # newcmp = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])

            # color palette 3
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.clf()
            plt.imshow(heatmap.T, cmap='YlGnBu_r', extent=extent, origin='lower', vmin=0, vmax=25)  #
            plt.colorbar()
            plt.title('image size distribution')
            plt.savefig('image_size_distribution.svg')
            plt.show()
            plt.clf()

        else:
            # # 2.sns custom version
            plt.figure(dpi=300)
            sns.heatmap(heatmap, annot=False, fmt='.1f', cbar=True, linewidths=.5, cmap="viridis",
                        xticklabels=xedges, yticklabels=yedges)  # cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            plt.title('image size distribution')  # 'image size distribution'
            # plt.savefig('../input/repeated_pattern_size_distribution.png')
            plt.show()
            plt.clf()

    else:
        # # 2. for img rp sizes
        w_edges = range(0, 380, 20)
        h_edges = range(0, 380, 20)

        w = np.array([int(i[0]) for i in sizes])
        h = np.array([int(i[1]) for i in sizes])
        heatmap, xedges, yedges = np.histogram2d(w, h, bins=(w_edges, h_edges))

        if sample:
            # # 1.express edition figure
            top = cm.get_cmap('Oranges_r', 128)
            bottom = cm.get_cmap('Blues', 128)
            newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
            newcmp = ListedColormap(newcolors, name='OrangeBlue')

            # newcmp = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.clf()
            plt.imshow(heatmap.T, cmap=newcmp, extent=extent, origin='lower')
            plt.title('image size distribution')
            # plt.savefig('../input/image_size_distribution.png')
            plt.show()
            plt.clf()

        else:
            # # 2.sns custom version
            plt.figure(dpi=300)
            sns.heatmap(heatmap, annot=False, fmt='.1f', linewidths=.5, cmap='Pastel2',
                        xticklabels=xedges, yticklabels=yedges)  # cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            plt.title('repeated pattern size distribution')  # 'image size distribution'
            # plt.savefig('../input/repeated_pattern_size_distribution.png')
            plt.show()
            plt.clf()


class DataAnalyze(object):
    """the size [w, h]"""
    def __init__(self, path, csvname):
        self.path = path
        self.csvname = csvname

    def save_size2csv(self):
        # # 1.get imgs sizes and save into a .csv file.
        imgs_w = []
        imgs_h = []
        imgs_size = []
        for curDir, dirs, files in os.walk(self.path):
            for file in files:
                img = cv2.imread(self.path + file)
                img_h = img.shape[1]
                imgs_h.append(img_h)
                img_w = img.shape[0]
                imgs_w.append(img_w)
                imgs_size.append([img_w, img_h])

        f = open(self.csvname, 'w')
        writer = csv.writer(f)
        for i in imgs_size:
            writer.writerow(i)
        info = ['min_w:', min(imgs_w), 'max_w:', max(imgs_w), 'min_h:', min(imgs_h), 'max_h:', max(imgs_h)]
        writer.writerow(info)

    def save_rpsize2csv(self):
        # # 1.get imgs' rp sizes and save into a .csv file.
        Dw = []
        Dh = []
        DD = []
        for curDir, dirs, files in os.walk(self.path):
            for dir in dirs:
                if dir[0] == 't':
                    for imgnames in os.listdir(self.path + dir):
                        dh = int(imgnames.split('_')[-1][:-4])
                        Dh.append(dh)
                        dw = int(imgnames.split('_')[-2])
                        Dw.append(dw)
                        DD.append([dw, dh])

        f = open(self.csvname, 'w')
        writer = csv.writer(f)
        for i in DD:
            writer.writerow(i)
        info = ['min_dw:', min(Dw), 'max_dw:', max(Dw), 'min_dh:', min(Dh), 'max_dh:', max(Dh)]
        writer.writerow(info)

    def read_csv(self):
        # # 2.read the.csv file, and get imgs sizes to a list
        self.sizes = []
        with open(self.csvname, 'r') as g:
            csvfile = csv.reader(g)
            csvfile = list(csvfile)
            self.head = csvfile[-1]
            csvfile2 = csvfile[:-1]
            for row in csvfile2:
                self.sizes.append(row)
        return self.sizes, self.head


path = '../../doc/2-input/'
csvname = 'dataset_image_sizes.csv'
data = DataAnalyze(path, csvname)
# data.save_rpsize2csv()
# 1. get data
data.save_size2csv()
# 2. read data
sizes, head = data.read_csv()
print(head)
# if 'save_size2csv', n = '1'; else, n='2'.

# 3. creat heatmap; sample true=matplotlib, otherwise=sns.
heatmap(sizes, sample=True, n='1', head=head)
# draw_scatter(sizes)







