# -*- coding:utf-8 -*-
import os
import time
import pandas as pd
import setting
import gc
import cv2
import numpy as np
# from DesignElementsExtraction import ForegroundElementsExtraction as FEE
# from DesignElementsExtraction import RedundantElementsRemoval as RER
# from DesignElementsVectorization import DesignElementVectorization as DEV
# from EditableDesignGeneration import EditableDesignGeneration as EDG
# from utils.results2excel import save_res_csv

from seg_methods.adaptive_threshod_segmentation import adaptive_threshod_segmentation as ad_thre_seg
from seg_methods.kmeans_segmentation import kmeans_seg
from seg_methods.meanshift_seg import meanshift_seg


def adaptive_thre_seg():
    savepath = setting.SegPath + 'thre_seg/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # res_name = setting.SavePath + 'input_cropedpattern_results.txt'
    # title = 'image_name, element_extraction_time, element_selection_time, element_vectorization_time, time_cost_all, num_elements, elements_info'
    # with open(res_name, "a") as f:
    #     f.write(title + '\n')
    #     f.close()

    infos = []
    type_ = 'adaptive'
    for files in os.listdir('./' + setting.ImgPath):
        try:
            start = time.time()
            file_path = './' + setting.ImgPath + files
            img_original = cv2.imread(file_path, 0)

            if type_ == 'adaptive':
                labels_ada_mean, labels_ada_gaussian = ad_thre_seg(files, img_original, type_, savepath)
                time_cost = time.time() - start

                info = [files, time_cost, labels_ada_mean, labels_ada_gaussian]
                infos.append(info)

            elif type_ == 'global':
                labels = ad_thre_seg(files, img_original, type_, savepath)
                time_cost = time.time() - start

                info = [files, time_cost, labels]
                infos.append(info)


        except Exception as E:
            print("Erro: " + files)
            with open(savepath + "ErroList.txt", "a") as f:
                f.write(str(files) + '\n')
        continue

    if type_ == 'adaptive':
        frame = pd.DataFrame(infos,
                             columns=['img_name', 'time_cost', 'labels_ada_mean', 'labels_ada_gaussian'])
        # print(frame)
        frame.to_excel(savepath + "seg_info_0604.xlsx")

    elif type_ == 'global':
        frame = pd.DataFrame(infos,
                             columns=['img_name', 'time_cost', 'labels'])
        # print(frame)
        frame.to_excel(savepath + "adap_thre_seg_info_0604.xlsx")


def kmeans():
    savepath = setting.SegPath + 'kmeans/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    k = 2
    infos = []
    for files in os.listdir('./' + setting.ImgPath):
        # try:
            start = time.time()
            file_path = './' + setting.ImgPath + files
            img_original = cv2.imread(file_path)
            lables = kmeans_seg(files, img_original, k, savepath)
            time_cost = time.time() - start

            info = [files, time_cost, lables]
            print(info)
            infos.append(info)

        # except Exception as E:
        #     print("Erro: " + files)
        #     with open(savepath + "ErroList.txt", "a") as f:
        #         f.write(str(files) + '\n')
        # continue

    frame = pd.DataFrame(infos,
                         columns=['img_name', 'time_cost', 'labels'])
    # print(frame)
    frame.to_excel(savepath + "kmeans_seg_info_0604.xlsx")


def meanshift():
    savepath = setting.SegPath + 'meanshift/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    infos = []
    for files in os.listdir('./' + setting.ImgPath):
        # try:
            start = time.time()
            file_path = './' + setting.ImgPath + files
            img_original = cv2.imread(file_path)
            lables = meanshift_seg(files, img_original, savepath)
            time_cost = time.time() - start

            info = [files, time_cost, lables]
            print(info)
            infos.append(info)

        # except Exception as E:
        #     print("Erro: " + files)
        #     with open(savepath + "ErroList.txt", "a") as f:
        #         f.write(str(files) + '\n')
        # continue

    frame = pd.DataFrame(infos,
                         columns=['img_name', 'time_cost', 'labels'])
    # print(frame)
    frame.to_excel(savepath + "meanshift_seg_info_0604.xlsx")


def main():
    # 1. seg experiment 1
    adaptive_thre_seg()
    # 2. seg experiment 2
    kmeans()
    # 3. seg experiment 3
    meanshift()


if __name__ == '__main__':
    main()