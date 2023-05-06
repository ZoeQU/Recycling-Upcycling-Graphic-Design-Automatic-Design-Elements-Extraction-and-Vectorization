# -*- coding:utf-8 -*-
import os

path_imgs = 'doc/2-input/'
for files in os.listdir(path_imgs):
    print(files)
    img_path = os.path.join(path_imgs, files)

    with open("2-input-file.txt", "a") as f:
        f.write(str(img_path) + '\n')
