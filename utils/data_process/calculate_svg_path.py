# -*- coding:utf-8 -*-
import lxml.etree as ET
from xml.dom import minidom
import os
from collections import defaultdict
from tqdm import trange


def parse_svg(fileName):
    tree = ET.parse(open(fileName, 'r'))
    root = tree.getroot()
    height = root.attrib['height']
    width = root.attrib['width']
    viewBox = root.attrib['viewBox']
    header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
             'width="' + str(width) + '" height="' + str(height) + '" viewBox="' + str(viewBox) + '"' + '\n' \
             'preserveAspectRatio = "xMidYMid meet" >'
    doc = minidom.parse(fileName)  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()
    return len(path_strings)


def rename_save_svginfo(curPath):
    info = defaultdict(list)
    for dir in os.listdir(curPath):
        tmp = dir.split('.')
        if len(tmp) <= 1:
            svgPath = os.path.join(curPath, dir + '/svg/')
            for svgName in os.listdir(svgPath):
                strs = svgName.split('_')

                deName = ''
                for i in strs[:-2]:
                    deName += i + '_'
                deName = deName + strs[-2]

                new_num = parse_svg(os.path.join(svgPath, svgName)) + 1
                temp_key = dir + '/' + str(deName)
                if temp_key in info.keys():
                    info[temp_key].append(new_num)
                else:
                    info.setdefault(temp_key, [new_num])

                newName = ''
                for a in strs[:-1]:
                    newName += (a + '_')
                newName = newName + str(new_num) + '.svg'
                os.rename(os.path.join(svgPath, svgName), os.path.join(svgPath, newName))
    return info