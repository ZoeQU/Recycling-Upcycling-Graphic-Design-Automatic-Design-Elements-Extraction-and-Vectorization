# -*- coding:utf-8 -*-
import os
import cairosvg
import cv2
import lxml.etree as ET
import pandas as pd
from PIL import Image, ImageDraw, ImagePath
import re
import matplotlib.pyplot as plt


def Hex2Rgb(value):
    """将16进制Hex 转化为 [R,G,B]"""
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def generate_svg(width, height, viewBox, filename, svgPaths):
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

        for k in svgPaths:
            svg.write(k + '\n')
        svg.write("</svg>")
    svg.close()


def update_svg(svg_file_name, temp_name):
    # Read the SVG file contents as bytes
    with open(svg_file_name, 'rb') as f:
        svg_data = f.read()

    # Remove the XML declaration from the SVG file
    svg_data = re.sub(rb'<\?xml.*\?>', b'', svg_data)

    # Parse the SVG data as an ElementTree object
    svg_file = ET.fromstring(svg_data)

    # Get the viewBox attribute and size of the SVG
    view_box = svg_file.get('viewBox')
    svg_size = tuple(map(int, view_box.split()[2:]))
    width = view_box.split()[2]
    height = view_box.split()[3]

    svg_paths = []
    # Loop through all the paths in the SVG and draw them on the image
    for path_elem in svg_file.findall('.//{http://www.w3.org/2000/svg}path'):
        path_ = path_elem.get('d')
        path_list = [round(float(coord)) for coord in re.findall(r'-?\d+\.?\d*', path_)]
        y = path_list[1::2]
        x = path_list[::2]
        coords = ''
        for i in range(len(x)):
            coords += str(x[i]) + ',' + str(y[i]) + ' '

        path_strings = 'M ' + coords + 'z'

        fill_color = path_elem.get('style').split(';')[0].split(':')[1]
        path_i = '<path d="' + path_strings + '" transform="translate(0,0) scale(1,1)" stroke="none" fill="' + fill_color + '"/>\n'
        svg_paths.append(path_i)

    # Save the updated svg
    generate_svg(width, height, view_box, temp_name, svg_paths)


def comparison(image1, image2):
    """
    compare two small images' similarity in pixel level
    :param image1: ori raster img
    :param image2: vector img that is transformed to raster one
    :return: similarity between them
    """
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the two images
    difference = cv2.absdiff(gray1, gray2)

    # Calculate the similarity score
    similarity = 1 - (difference.sum() / (255 * difference.size))

    # # Print the similarity score
    # print("Similarity: {:.2f}%".format(similarity * 100))
    return similarity


de_path = "../../doc/de-(design_elements)/"
# todo "../../doc/de-adobe-result/"
evaluate_paths = ["../../doc/de-adobe-result/"]
# evaluate_paths = ["../../doc/de-vectors/", "../../doc/de-VectorMagic-fully-automatic/"] || "../../doc/de-GTV/"


# 1. read ori vector image, get its size (h,w)
results = []
for evaluate_path in evaluate_paths:
    result_path = evaluate_path[:-1] + "-rasterization/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result = []
    for vector_de_name in os.listdir(evaluate_path):
        # try:
        #     tt = vector_de_name.split('_')
        #     raster_de_name = '_'.join(tt[:-3]) + '_' + tt[-3] + '.png'

            raster_de_name = vector_de_name[:-4] + '.png'  # for GTV, vectormagic

            vector_de_path = os.path.join(evaluate_path, vector_de_name)
            temp_name = result_path + vector_de_name[:-4] + ".png"

            cairosvg.svg2png(url=vector_de_path, write_to=temp_name)  # workable for regular svg

            # # !!! prepare workable svg for rasterization,(workable for GTV)
            # temp_svg_name = result_path + vector_de_name[:-4] + "_update.svg"
            # update_svg(vector_de_path, temp_svg_name)
            # cairosvg.svg2png(url=temp_svg_name, write_to=temp_name)

            tmp_img = cv2.imread(temp_name)
            plt.imshow(tmp_img)
            plt.show()
            plt.close()

            # 2. pair read raster image, convert to same size image
            raster_de_img = cv2.imread(os.path.join(de_path, raster_de_name))
            raster_de_size = raster_de_img.shape  #(h, w)
            re_size = (raster_de_size[1], raster_de_size[0])
            tempimg = cv2.imread(temp_name)  # svg-raster file
            vector_de_img_resized = cv2.resize(tempimg, re_size, interpolation=cv2.INTER_AREA)

            # 3. use Template Matching to compare their similarity
            similarity = comparison(raster_de_img, vector_de_img_resized)  # ori, vector
            print(similarity)
            result.append(similarity)
        # except Exception as e:
        #     pass
    results.append(result)

# 4. record results
# nn = ['current_study', 'VectorMagic']
nn = ['VectorMagic']
for i in range(len(results)):
    df = pd.DataFrame(results[i], columns=[nn[i]])
    savename = "../../doc/output/" + nn[i] + "_vector_fidelity_results.xlsx"
    df.to_excel(savename, index=False)






