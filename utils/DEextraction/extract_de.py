# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import setting
from shutil import copyfile
from sys import exit

aim_path = "doc/design_elements/"
if not os.path.exists(aim_path):
    os.makedirs(aim_path)

for curDir, dirs, files in os.walk('doc/output-final-0612/'):
    for i in dirs:
        de_path = os.path.join('doc/output-final-0612/', i, 'keep/')
        if os.path.exists(de_path):
            for j in os.listdir(de_path):
                ori_adress = os.path.join(de_path, j)
                copyfile(ori_adress, os.path.join(aim_path, j))

print('finish copy')