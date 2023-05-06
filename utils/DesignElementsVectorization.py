# -*- coding:utf-8 -*-
import math
import setting
from sklearn.cluster import KMeans
import time
import numpy as np
from random import randint as ri
import cv2  # 3.4.2
import imutils
import os
import cairosvg
from matplotlib import pyplot as plt
from DEvectorization.cal_functions import (color_blocks, cal_rectangle, cal_centroid,
                                           cal_points_distance, cal_area, cal_pixel_distribution,
                                           color_his, ColourDistance, RGB_to_Hex, Hex2Rgb)
from DEvectorization.vectroizationfunctions import temp_svg, vectorize

import warnings
warnings.filterwarnings("ignore")


def generate_svg(width, height, viewBox, filename, bkrecg, Path_order):
    # 生成新的vector graphic
    header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
             'width="' + str(width) + '" height="' + str(height) + '" viewBox="' + str(viewBox) + '"' + '\n' \
             'preserveAspectRatio = "xMidYMid meet" >'

    with open(filename, 'w') as svg:
        svg.write('<?xml version="1.0" standalone="no"?>\n'
                  '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                  ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
        svg.write(header + '\n')
        svg.write('<metadata>\n'
                  'Created by ZoeQU, 2022\n'
                  '</metadata>\n')
        svg.write(bkrecg + '\n')  # add rectangle in bk color
        for k in Path_order:
            svg.write(k[1] + '\n')
        svg.write("</svg>")
    svg.close()


def DesignElementVectorization(imname, bk_color, file, paths, visualization):
    """
    矢量化筛选后的elements
    :param imname: element图片名
    :param bk_color: 背景颜色
    :param file: 当前实验的图片名，作为字符串用来保存各个步骤的图片
    :param visualization: 是否存放一些步骤图片，默认true
    :return:
    """
    SavePath, processPath, elementsPath, svgPath, keepPath = paths

    start_time = time.time()
    im = cv2.imread(keepPath + imname)

    if im is not None:
        im_size = (im.shape[0], im.shape[1])  # w,h
        if im.shape[0] < 100 and im.shape[1] < 100:
            im = cv2.resize(im, (2 * im.shape[1], 2 * im.shape[0]), interpolation=cv2.INTER_CUBIC)

        im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)  # Denoise

        """adaptive calculate 'k'"""
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        k = color_his(img_gray, imname, processPath, visualization)

        """k-means for color reduction"""
        temp_im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # convert to LAB
        temp_in = temp_im.reshape((-1, 3))
        Kcolor = KMeans(n_clusters=k)
        Kcolor.fit(temp_in)
        labels = Kcolor.predict(temp_in)
        cluster_colors = Kcolor.cluster_centers_

        img0 = np.ones((2, 2), dtype=np.uint8)
        rgb_img = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)  # convert to rgb
        cluster_rgb = []
        for i in cluster_colors:
            h, s, v = (i[0], i[1], i[2])
            rgb_img[:, :, :] = (h, s, v)
            RGB = cv2.cvtColor(rgb_img, cv2.COLOR_LAB2RGB)  # or cv2.COLOR_HSV2RGB)
            cluster_rgb.append(RGB[1, 1, :].tolist())

        """比较颜色，找出bk_color"""
        distance_value = []
        for color_i in cluster_rgb:
            distance_value.append([color_i, ColourDistance(bk_color, color_i)])
        distance_value = sorted(distance_value, key=lambda x: x[1])
        a = distance_value[0][0]
        ind = cluster_rgb.index(a)
        bk_color_2 = [bk_color, cluster_rgb[ind]]
        if visualization:
            color_blocks(bk_color_2, processPath + imname[:-4] + '_bk_2colors.png')
        # bk_color = cluster_rgb[ind]
        cluster_rgb[ind] = bk_color

        """show color blocks"""
        if visualization:
            color_blocks(cluster_rgb, processPath + imname[:-4] + '_color_block.png')

        """separate by color"""
        CC = []
        IMGS = []
        for i in np.unique(labels):
            cc = cluster_rgb[i]  # rgb
            img_af_cluster = labels.copy()
            if cc != [0, 0, 0]:
                img_af_cluster = np.where(img_af_cluster == i, np.array(cluster_rgb[i]).reshape(3, 1),
                                          np.array([0, 0, 0]).reshape(3, 1))
            else:
                img_af_cluster = np.where(img_af_cluster == i, np.array(cluster_rgb[i]).reshape(3, 1),
                                          np.array([255, 255, 255]).reshape(3, 1))

            r = np.reshape(img_af_cluster[0], (im.shape[0], im.shape[1]))
            g = np.reshape(img_af_cluster[1], (im.shape[0], im.shape[1]))
            b = np.reshape(img_af_cluster[2], (im.shape[0], im.shape[1]))
            merged = cv2.merge([r, g, b])  # 3D color image
            IMGS.append([cc, merged])

            if visualization:
                plt.imshow(merged)
                plt.axis('off')
                plt.savefig(processPath + imname[:-4] + 'colormask_' + str(ri(0, 9)) +'.png')
                plt.close('all')

            """为了得到 color_mask_binary"""
            temp_ = np.where(merged == cc, [0, 0, 0], [255, 255, 255])  # 3 channel black & white image
            GrayImage = cv2.cvtColor(temp_.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            ret, color_mask_binary = cv2.threshold(GrayImage, 50, 255, cv2.THRESH_BINARY)

            """Statistical pixel distribution"""
            try:
                CC_rectangle = cal_rectangle(color_mask_binary)
            except Exception:
                CC_rectangle = []
                continue
            # centroid = cal_centroid(color_mask_binary)
            centroid = (0, 0)
            # arearate = cal_area(merged)
            arearate = 0
            # boxrate = cal_pixel_distribution(merged, cc)
            boxrate = 0
            cc_satisic = [cc, centroid, arearate, CC_rectangle, boxrate, color_mask_binary]
            CC.append(cc_satisic)  # CC = [cc,centroid,arearate,CC_rectangle,boxrate,color_mask_binary]

        if visualization:
            plt.figure(figsize=(20, 8))
            for i, image in enumerate(IMGS):
                plt.subplot(2, 10, i+1)
                plt.imshow(image[1])
                plt.title(str(image[0]))
                plt.axis('off')
            plt.savefig(processPath + imname[:-4] + '_kmean.png')
            # plt.show()
            plt.close('all')

        # svg文件的必要head信息
        width = str(im.shape[1]) + '.000000pt'
        height = str(im.shape[0]) + '.000000pt'
        viewBox = '0 0 ' + width[:-2] + ' ' + height[:-2]
        # 生成“背景”svg文件，后又栅格化成png文件用以统计pix_value=0 的像素值数目（原因为，svg与png相互转换后图片尺寸不一致）
        BKcolor = RGB_to_Hex(bk_color)
        bkrecg = '<rect width="' + str(width[:-2]) + '" height="' + str(height[:-2]) + '" fill="' + str(BKcolor) + '"/>'
        temppng = processPath + '/' + imname[:-4] + '_BK.png'
        tempsvg = temp_svg(bkrecg, width, height, viewBox, processPath)
        cairosvg.svg2png(url=tempsvg, write_to=temppng)
        temp_ = cv2.imread(temppng)
        a_num = np.where(temp_ == bk_color)
        A = len(a_num[0])
        os.remove(tempsvg)  # 删除临时 bk.svg 文件
        os.remove(temppng)  # 删除临时 bk.png 文件

        # 逐个颜色mask矢量化
        pathes = []
        for q in range(len(CC)):
            if CC[q][0] != bk_color:  # 判断是否是背景颜色， 可以精简矢量化后path的数目
                color_mask_bmp = processPath + imname[:-4] + '_color_' + str(q) + '.bmp'
                cv2.imwrite(color_mask_bmp, CC[q][5])
                rgb_value = CC[q][0]
                path_strings, fillcolor = vectorize(color_mask_bmp, rgb_value)

                if len(path_strings) > 0:
                    for j in range(len(path_strings)):
                        path_i = '<path d="' + str(path_strings[j]) + '" transform="translate(0.000000,' + str(
                            height[:-2]) + ')' \
                           ' scale(0.100000,-0.100000)" stroke="none" ' \
                           'fill="' + fillcolor + '"/>\n'
                        pathes.append(path_i)
                os.remove(color_mask_bmp)
                os.remove(color_mask_bmp[:-4] + '.svg')

        """order path by area, small->large| path_area by calculate black pixel"""
        p_order = []
        p_ori = []
        for pi in pathes:
            tempname = temp_svg(pi, width, height, viewBox, processPath)  # pi is path
            temp = processPath + '/' + 'temp .png'
            cairosvg.svg2png(url=tempname, write_to=temp)
            temp_ = cv2.imread(temp)

            """count Black pixels"""
            h, w, _ = temp_.shape
            npim = temp_[:, :, 0] + temp_[:, :, 1] + temp_[:, :, 2]
            black_pixel_num = len(npim[npim == 0])
            content = npim.shape[0] * npim.shape[1] - black_pixel_num
            # record ori path
            p_ori.append([content, pi])

            # record path after filter
            if content > (0.002 * A):   # meaningless path的自定义标准
                p_order.append([content, pi])
            else:
                pass
            os.remove(tempname)
            os.remove(temp)

        Path_ori = sorted(p_ori, key=lambda x: x[0])
        num_ori = len(Path_ori) + 1
        Path_order = sorted(p_order, key=lambda x: x[0])
        num_af_filter = len(Path_order) + 1

        # # 生成新的vector graphic
        filename_ori = svgPath + imname[:-4] + '_ori_num_' + str(num_ori) + '.svg'
        generate_svg(width, height, viewBox, filename_ori, bkrecg, Path_ori)

        filename_filter = svgPath + imname[:-4] + '_filter_num_' + str(num_af_filter) + '.svg'
        generate_svg(width, height, viewBox, filename_filter, bkrecg, Path_order)

        timecost = round(time.time() - start_time, 1)
        return [im_size, filename_ori, num_ori, filename_filter, num_af_filter, timecost]

    else:
        return None
