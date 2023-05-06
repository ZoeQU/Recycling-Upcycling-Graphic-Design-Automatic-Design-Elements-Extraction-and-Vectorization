# -*- coding:utf-8 -*-
import cv2  # 3.4.2
import numpy as np


class Img2svg(object):
    """
    将 image（numpy 数组）转换成svg文件
    """
    def __init__(self, image, savePath):
        self.image = image
        self.savePath = savePath

    def preprocess(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.pathes = []
        for cont in contours[1]:
            epsilon = 0.01 * cv2.arcLength(cont, True)  # 取轮廓长度的1%为epsilon
            box = cv2.approxPolyDP(cont, epsilon, True)  # 预测多边形
            box = np.reshape(box, (-1, 2))
            start = ' '.join(str(i) for i in box[0])
            t = ''
            for k in box[1:]:
                for j in k:
                    t += ' ' + str(j)
            P = 'M' + start + ' L' + t + 'z'
            path_i = '<path d="' + str(
                P) + '"' + ' transform="translate(0, 0)' + ' scale(1, 1)"' + ' stroke="green"' + ' fill=' + '"none"' + '/>' + '\n'
            self.pathes.append(path_i)
            self.img = cv2.polylines(image, [box], True, (0, 0, 255), 10)
        cv2.imwrite('test.png', self.img)

        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.viewBox = '0 0 ' + str(self.width) + ' ' + str(self.height)

    def temp_svg(self):
        header = '<svg version="1.0" xmlns="http://www.w3.org/2000/svg" \n' \
                 'width="' + str(self.width) + '" height="' + str(self.height) + \
                 '" viewBox="' + str(self.viewBox) + '"' + '\n' \
                'preserveAspectRatio = "xMidYMid meet" >'
        tempname = self.savePath + 'temp.svg'
        with open(tempname, 'w') as svg:
            svg.write('<?xml version="1.0" standalone="no"?>\n'
                      '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"\n'
                      ' "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n')
            svg.write(header + '\n')
            for path_i in self.pathes:
                svg.write(path_i + '\n')
            svg.write("</svg>")
        svg.close()

    def run(self):
        self.preprocess()
        self.temp_svg()


imgpath = 'test.jpg'
image = cv2.imread(imgpath)
svgfile = Img2svg(image, '')
svgfile.run()




