# -*- coding:utf-8 -*-
import math
from random import randint as ri
from random import uniform as rf
import lxml.etree as ET
from xml.dom import minidom
import time
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


def read_svg(svg_name):
    """
    minidom.parse 和 ET.parse(open(svg_name, 'r'))
    两种不同的读取svg文件的方法，
    侧重点不同，因此用两种方式读取
    :param svg_name:
    :return:
    """
    tree = ET.parse(open(svg_name, 'r'))
    root = tree.getroot()
    height = root.attrib['height']
    width = root.attrib['width']
    height = 10 * int(height.split('.')[0])
    width = 10 * int(width.split('.')[0])
    # viewBox = root.attrib['viewBox']
    viewBox = '0 0 ' + str(width) + ' ' + str(height)
    header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n'\
             'width="' + str(width) + '" height="' + str(height) + '" viewBox="' + str(viewBox) + '"' + '\n' \
             'preserveAspectRatio = "xMidYMid meet" >'

    doc = minidom.parse(svg_name)
    rect_color = [rect.getAttribute('fill') for rect in doc.getElementsByTagName('rect')]
    path = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    doc.unlink()

    return path, rect_color, header, height, width


def read_svg_as_txt(svg_name):
    """
    无意义的读取而不解析，无法进行后续处理
    :param svg_name:
    :return:
    """
    with open(svg_name, 'r') as f:
        data = f.readlines()
        for i in data:
            print(i)


def hex_random():
    """生成一个随机的hex值"""
    # color: int
    color1 = ri(0, 255)
    color2 = ri(0, 255)
    color3 = ri(0, 255)
    color1 = hex(color1)
    color2 = hex(color2)
    color3 = hex(color3)
    ans = "#" + color1[2:] + color2[2:] + color3[2:]
    return ans


def random_bk_rect(rect_color, height, width):
    bk_color = hex_random()
    bk_rect = '<rect width="' + str(width) + '" height="' + str(height) + '" fill="' + str(bk_color) + '"/>'
    return bk_rect, bk_color


def generate_svg(path_i, header, svgname, processPath):
    """"""
    tempname = processPath + svgname + '_new.svg'
    with open(tempname, 'w') as svg:
        svg.write('<?xml version="1.0" standalone="no"?>\n'
                  '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                  ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
        svg.write(header + '\n')
        svg.write('<metadata>\n'
                  'Written by ZoeQu 2022\n'
                  '</metadata>\n')
        svg.write(path_i + '\n')
        svg.write("</svg>")
    svg.close()
    return tempname


def random_process_path(path, height, width):
    fillcolor = hex_random()
    translate_param = (ri(0, int(width/10)), ri(0, int(height/10)))
    rotate_param = (ri(0, 90), int(width/2), int(height / 2))
    skewX_param = (ri(0, 10))
    skewY_param = (ri(0, 10))
    scale_param = (round(rf(0.8, 1), 2), round(rf(0.8, 1), 2))
    # 'trasnlate' + str(translate_param) +
    path_new = '<path d="' + str(path) + '"' + ' transform="' + 'scale' + \
               str(scale_param) + 'rotate' + str(rotate_param) + 'skewX(' + str(skewX_param) +\
               ')" stroke="none" ' + 'fill="' + fillcolor + '"/>\n'
    return path_new


def generate_new_svg(savename, header, pathes_new, new_bk_rect):
    with open(savename, 'w') as svg:
        svg.write('<?xml version="1.0" standalone="no"?>\n'
                  '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                  ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
        svg.write(header + '\n')
        svg.write('<metadata>\n'
                  'Created by ZoeQu, written in 2022\n'
                  '</metadata>\n')
        svg.write(new_bk_rect + '\n')
        for k in pathes_new:
            svg.write(k + '\n')
        svg.write("</svg>")
    svg.close()


