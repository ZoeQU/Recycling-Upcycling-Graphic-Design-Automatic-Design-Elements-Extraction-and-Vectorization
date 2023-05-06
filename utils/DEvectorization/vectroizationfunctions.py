# -*- coding:utf-8 -*-
import math
import setting
import lxml.etree as ET
from xml.dom import minidom
import time
from collections import defaultdict
import argparse
import sys
from operator import itemgetter
import itertools
from collections import defaultdict, Counter
import numpy as np
import cv2  # 3.4.2
import imutils
import os
import cairosvg
from matplotlib import pyplot as plt
from utils.DEvectorization.cal_functions import (RGB_to_Hex, ColourDistance,
                                                 cal_points_distance, cal_area,
                                                 cal_centroid, cal_pixel_distribution,
                                                 cal_rectangle, Hex2Rgb)
import warnings
warnings.filterwarnings("ignore")

def merge_img(im_A, im_B):
    mergerd_bycolor = im_A + im_B
    mergerd_bycolor = np.where([[c == 255 or c == 0 for c in row] for row in mergerd_bycolor], 0, 255)
    cv2.imwrite(setting.SavePath+'temp_merged.png', mergerd_bycolor)
    mergerd_bycolor = cv2.imread(setting.SavePath+'temp_merged.png')
    GrayImage = cv2.cvtColor(mergerd_bycolor, cv2.COLOR_BGR2GRAY)
    ret, temp_merged_binary = cv2.threshold(GrayImage, 50, 255, cv2.THRESH_BINARY)
    os.remove(setting.SavePath+'temp_merged.png')
    return temp_merged_binary


def merge(CC_list, thre):
    if len(CC_list) <= 2:
        new_rgb = [x[0] for x in CC_list]
        return CC_list, new_rgb
    else:
        compare_list = list(itertools.combinations(CC_list, 2))
        cc_b = []  # CC = [0-cc,1-centroid,2-arearate,3-CC_rectangle,4-boxrate,5-merged_binary]
        New = []
        for j in compare_list:
            color_distance = ColourDistance(j[0][0], j[1][0])
            centroid_distance = cal_points_distance(j[0][1], j[1][1])
            cc_b.append([color_distance, centroid_distance, j])
        removelist = []
        for d in cc_b:
            if d[0] < thre:  # and d[1] < 20
                im_merged_bi = merge_img(d[2][0][5], d[2][1][5])
                # # compare area to keep color # #
                if d[2][0][2] > d[2][1][2]:
                    keep_cc = d[2][0][0] # keep_c is color of 'img after merged'
                else:
                    keep_cc = d[2][1][0]
                new_centroid = cal_centroid(im_merged_bi)
                new_arearate = cal_area(im_merged_bi)
                new_boxrate = cal_pixel_distribution(im_merged_bi, 0)
                new_cc_rectangle = cal_rectangle(im_merged_bi)
                add_CC = [keep_cc, new_centroid, new_arearate, new_cc_rectangle, new_boxrate, im_merged_bi]
                New.append(add_CC)
                removelist.append(d[2][0])
                removelist.append(d[2][1])
        # # remove duplicate ones
        temps1 = list_duplicates(removelist)
        for jj in temps1:
            delone = removelist[jj[1][0]]
            CC_list.remove(delone)
        # # merge duplicate ones
        temps2 = list_duplicates(New)
        CC2 = []
        for ii in temps2:
            merge1 = []
            if len(ii[1]) > 0:
                merge1.append([New[x][5] for x in ii[1]])
                merge2 = np.sum(merge1[0],axis=0)
                pixmax = np.max(merge2)
                mergerd_bycolor = np.where([[c == pixmax for c in row] for row in merge2], 255, 0)
                cv2.imwrite(setting.SavePath + 'temp_merged.png', mergerd_bycolor)
                mergerd_bycolor = cv2.imread(setting.SavePath + 'temp_merged.png')
                GrayImage = cv2.cvtColor(mergerd_bycolor, cv2.COLOR_BGR2GRAY)
                ret, temp_merged_binary = cv2.threshold(GrayImage, 50, 255, cv2.THRESH_BINARY)
                new_centroid = cal_centroid(temp_merged_binary)
                new_arearate = cal_area(temp_merged_binary)
                new_boxrate = cal_pixel_distribution(temp_merged_binary, 0)
                new_cc_rectangle = cal_rectangle(temp_merged_binary)
                keep_cc = Hex2Rgb(ii[0])
                add_CC = [keep_cc, new_centroid, new_arearate, new_cc_rectangle, new_boxrate, temp_merged_binary]
                CC2.append(add_CC)
                os.remove(setting.SavePath + 'temp_merged.png')
            else:
                CC2.append(New[ii[1][0]])
    if len(CC_list) > 0:
        for x in CC2:
            CC_list.append(x)
    else:
        CC_list = CC2
    new_rgb = [x[0] for x in CC_list]
    return CC_list, new_rgb


def list_duplicates(seq):
    tempList = []
    for i in range(len(seq)):
        rgb1_hex = RGB_to_Hex(seq[i][0])
        tempList.append([seq[i], rgb1_hex])
    tally = defaultdict(list)
    for i, item in enumerate(tempList):
        tally[item[1]].append(i)
    return tally.items()
    # return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def temp_svg(path_i, width, height, viewBox, processPath):
    """
    根据输入的svg path信息生成一个svg文件
    :param path_i: svg路径信息
    :param width: svg文件head信息
    :param height: svg文件head信息
    :param viewBox: svg文件head信息
    :param processPath: 存放地址文件夹
    :return: svg文件
    """
    header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
             'width="' + str(width) + '" height="' + str(height) + \
             '" viewBox="' + str(viewBox) + '"' + '\n' \
            'preserveAspectRatio = "xMidYMid meet" >'
    tempname = processPath + 'temp.svg'
    with open(tempname, 'w') as svg:
        svg.write('<?xml version="1.0" standalone="no"?>\n'
                  '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                  ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
        svg.write(header + '\n')
        svg.write(path_i + '\n')
        svg.write("</svg>")
    svg.close()
    return tempname


def vectorize(img_name, rgb_value):
    """
    利用protrace.exe矢量化
    :param img_name: 图片名
    :param rgb_value: 矢量化后填充的颜色值（Hex格式）
    :return:
    """
    # os.system('cd /home/user/0-zoe_project/ImageVectorization_line_vector/')
    # os.system('potrace binary.bmp -b svg')
    os.system('potrace ' + img_name + ' -b svg')
    fillcolor = RGB_to_Hex(rgb_value)
    tree = ET.parse(open(img_name[:-4] + '.svg', 'r'))
    root = tree.getroot()
    height = root.attrib['height']
    width = root.attrib['width']
    viewBox = root.attrib['viewBox']
    header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
             'width="' + str(width) + '" height="' + str(height) + '" viewBox="' + str(viewBox) + '"' + '\n' \
             'preserveAspectRatio = "xMidYMid meet" >'
    doc = minidom.parse(img_name[:-4] + '.svg')  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()

    return path_strings, fillcolor