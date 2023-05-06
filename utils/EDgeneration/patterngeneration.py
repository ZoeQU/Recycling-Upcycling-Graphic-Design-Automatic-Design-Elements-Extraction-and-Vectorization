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


def rowgroup(pathes_new, width):
    rowgroup = []
    for i in range(5):
        grouppath = '<g id="' + str(i) + '" ' + 'transform=' + '"translate(' + str(i * width) + ',0)' + 'scale(0.8,0.8)' + '">' + '\n'
        rowgroup.append(grouppath)
        for j in pathes_new:
            rowgroup.append(j)
        rowgroup.append('</g>' + '\n')
    return rowgroup


def colgroup(pathes_new, height):
    colgroup = []
    for i in range(5):
        grouppath = '<g id="' + str(i) + '" ' + 'transform=' + '"translate(0,' + str(i * height) + ')' + 'scale(0.8,0.8)' + '">' + '\n'
        colgroup.append(grouppath)
        for j in pathes_new:
            colgroup.append(j)
        colgroup.append('</g>' + '\n')
    return colgroup


def patterngeneration(savename, type, header, patternwidth, patternheight, new_bk_rect, pathes_new):
    row = rowgroup(pathes_new, patternwidth / 4)
    col = colgroup(pathes_new, patternheight / 4)

    with open(savename, 'w') as svg:
        svg.write('<?xml version="1.0" standalone="no"?>\n'
                  '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                  ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
        svg.write(header + '\n')
        svg.write('<metadata>\n'
                  'Created by ZoeQu, written in 2022\n'
                  '</metadata>\n')
        svg.write(new_bk_rect + '\n')
        if type == 'straight':
            for r in range(4):
                svg.write('<g id="row' + str(r) + '" ' + 'transform=' + '"translate(' + str(int(patternwidth / 30)) + ',' + str(r * patternheight / 4) + ')' + '">' + '\n')
                for k in row:
                    svg.write(k + '\n')
                svg.write('</g>' + '\n')

        if type == 'tile':
            for r in range(4):
                if (r % 2) == 0:
                    svg.write('<g id="row' + str(r) + '" ' + 'transform=' + '"translate(' + str(int(patternwidth / 30)) + ',' + str(r * patternheight / 4) + ')' + '">' + '\n')
                else:
                    svg.write('<g id="row' + str(r) + '" ' + 'transform=' + '"translate(' + str(int(patternwidth / 30) - int(patternwidth / 8)) + ',' + str(r * patternheight / 4) + ')' + '">' + '\n')
                for k in row:
                    svg.write(k + '\n')
                svg.write('</g>' + '\n')

        if type == 'half-drop':
            for c in range(4):
                if (c % 2) == 0:
                    svg.write('<g id="col' + str(c) + '" ' + 'transform=' + '"translate(' + str(int(c * patternwidth / 4)) + ', ' + str(int(patternheight / 30)) + ')' + '">' + '\n')
                else:
                    svg.write('<g id="col' + str(c) + '" ' + 'transform=' + '"translate(' + str(int(c * patternwidth / 4)) + ', ' + str(int(patternheight / 30) - patternheight / 8) + ')' + '">' + '\n')
                for k in col:
                    svg.write(k + '\n')
                svg.write('</g>' + '\n')

        svg.write("</svg>")
    svg.close()