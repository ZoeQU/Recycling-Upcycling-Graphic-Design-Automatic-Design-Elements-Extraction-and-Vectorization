# -*- coding:utf-8 -*-
import lxml.etree as ET
from xml.dom import minidom
import time
import os
import csv


# tree = ET.parse(open(img_name[:-4] + '.svg', 'r'))
# root = tree.getroot()
# height = root.attrib['height']
# width = root.attrib['width']
# viewBox = root.attrib['viewBox']
# header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
#          'width="' + str(width) + '" height="' + str(height) + '" viewBox="' + str(viewBox) + '"' + '\n' \
#                                                                                                     'preserveAspectRatio = "xMidYMid meet" >'
# doc = minidom.parse(img_name[:-4] + '.svg')  # parseString also exists
# path_strings = [path.getAttribute('d') for path
#                 in doc.getElementsByTagName('path')]
# doc.unlink()

def parse_adobe():
    adobe = []
    for file in os.listdir('adobe/'):
        doc = minidom.parse('adobe/' + file)  # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()
        num = len(path_strings)
        adobe.append([file, num])

    adobe_name = 'adobe_svg_info.csv'
    title = ['image name, adobe']
    with open(adobe_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(title)
        writer.writerows(adobe)


def parse_gtv():
    gtv = []
    for file in os.listdir('gtv/'):
        doc = minidom.parse('gtv/' + file)  # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()
        num = len(path_strings)
        gtv.append([file, num])

    gtv_name = 'gtv_svg_info.csv'
    title = ['image name, gtv']
    with open(gtv_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(title)
        writer.writerows(gtv)


def parse_ours():
    ours = []
    for file in os.listdir('ours/'):
        doc = minidom.parse('ours/' + file)  # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()
        num = len(path_strings)
        ours.append([file, num])

    ours_name = 'our_svg_info.csv'
    title = ['image name, ours']
    with open(ours_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(title)
        writer.writerows(ours)


def parse_vectormagic():
    vectormagic = []
    for file in os.listdir('vectormagic/'):
        doc = minidom.parse('vectormagic/' + file)  # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()
        num = len(path_strings)
        vectormagic.append([file, num])

    ours_name = 'vectormagic_svg_info.csv'
    title = ['image name, vectormagic']
    with open(ours_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(title)
        writer.writerows(vectormagic)


def main():
    parse_adobe()
    parse_gtv()
    parse_ours()
    parse_vectormagic()



if __name__ == '__main__':
    main()
