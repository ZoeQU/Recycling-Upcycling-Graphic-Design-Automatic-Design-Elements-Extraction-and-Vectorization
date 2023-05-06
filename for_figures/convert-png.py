# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import cairosvg


for files in os.listdir('for_design/'):
    tempsvg = 'for_design/' + files
    temppng = 'for_design/' + files[:-4] + '.png'
    cairosvg.svg2png(url=tempsvg, write_to=temppng)

print('finish')