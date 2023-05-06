# -*- coding:utf-8 -*-
import os
import time
from random import randint as ri
import setting
from utils.EDgeneration.randomprocessfunctions import (read_svg, random_bk_rect,
                                                       random_process_path, generate_new_svg)
from utils.EDgeneration.patterngeneration import patterngeneration
import webbrowser
import cairosvg
import warnings
warnings.filterwarnings("ignore")

def EditableDesignGeneration(file, type):
    svgPath = '/home/user/0-zoe_project/ImgVectorization/' + setting.SavePath + file[:-4] + '/' + 'svg' + '/'
    generatePath = '/home/user/0-zoe_project/ImgVectorization/' + setting.SavePath + str(file[0:-4]) + '/' + 'generate' + '/'
    if not os.path.exists(generatePath):
        os.makedirs(generatePath)

    for svg in os.listdir(svgPath):
        # svg = 'media_shapes2_0_num_14.svg'
        # # 1. new element generation
        svg_name = svgPath + svg
        pathes, bk_color, svg_header, height, width = read_svg(svg_name)
        new_bk_rect, new_bk_color = random_bk_rect(bk_color, height, width)
        # print(new_bk_rect)
        pathes_new = []  # pathes group after random process
        for path in pathes:
            path_new = random_process_path(path, height, width)
            pathes_new.append(path_new)
        newelementname = generatePath + svg[:-4] + '_Newelement.svg'
        generate_new_svg(newelementname, svg_header, pathes_new, new_bk_rect)

        # # 2. new pattern generation
        patternwidth = 4 * width
        patternheight = 4 * height
        patternheader = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" width="' \
                        + str(patternwidth) + 'pt" height="' + str(patternheight) + \
                        'pt" viewBox="' + '0 0 ' + str(patternwidth) + ' ' + str(patternheight) + '" ' + \
                        'preserveAspectRatio = "xMidYMid meet" >'
        patternbk = '<rect width="' + str(patternwidth) + '" height="' + str(patternheight) + \
                    '" fill="' + str(new_bk_color) + '"/>'

        newpatternname = generatePath + svg[:-4] + '_' + str(type) + '_pattern.svg'
        patterngeneration(newpatternname, type, patternheader, patternwidth, patternheight, patternbk, pathes_new)


# """test"""
# EditableDesignGeneration(file='media_shapes2.jpg', type='half-drop')