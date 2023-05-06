import numpy as np
import cv2
import random
import os

# 随机在一张图像上截取任意大小图片
imp = 'doc/2-input/crop'
for file in os.listdir(imp):
    img = cv2.imread(os.path.join(imp, file))

    # h、w为想要截取的图片大小
    h = 250
    w = 250

    # 随机产生x,y   此为像素内范围产生
    y = random.randint(1, 50)
    x = random.randint(1, 50)
    # 随机截图
    cropImg = img[(y):(y + h), (x):(x + w)]
    cv2.imwrite('doc/2-input/' + file[:-4] + '_cropped.png', cropImg)





































