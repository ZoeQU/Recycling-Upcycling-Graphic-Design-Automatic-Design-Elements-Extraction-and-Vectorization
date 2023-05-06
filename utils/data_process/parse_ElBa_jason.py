# -*- coding:utf-8 -*-
import json
import os
import requests
# import torchvision.datasets as dset
# import torchvision.transforms as transforms


class GetJsonInfo:
    """extract ['images'] and ['annotations'] in .json file """
    def __init__(self, file_name):
        super(GetJsonInfo, self).__init__()
        self.file_name = file_name

    def read_json(self):
        file = open(self.file_name, 'r', encoding='utf-8')
        maps = []
        for line in file:
            maps.append(json.loads(line))

        imgs = []
        for i, img_ in enumerate(maps[0]['images']):
            imgs.append(img_)

        annotations = []
        for i, annotation in enumerate(maps[0]['annotations']):
            annotations.append(annotation)
        return imgs, annotations


def write_json(save_name, im_info, im_annotation):
    info = {"info": {"description": "TextonSee Dataset", "url": "---", "version": "2.0.0", "year": 2018,
                                    "contributor": "mgodi-cjoppi-nlanza", "date_created": "2019-03-09 10:26:29.511484"},
                      "licenses": [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike License",
                                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}],
                      "categories": [{"id": 1, "name": "circle", "supercategory": "macrotextons"},
                                    {"id": 2, "name": "line", "supercategory": "macrotextons"},
                                    {"id": 3, "name": "polygon", "supercategory": "macrotextons"}]}
    with open(save_name, 'w', encoding='utf-8') as f:
        for ii in im_info:
            info.setdefault('images', []).append(ii)
        for jj in im_annotation:
            info.setdefault('annotations', []).append(jj)
        f.write(json.dumps(info, ensure_ascii=False) + '\n')


def img_names(img_path):
    return os.listdir(img_path)


def get_img_info(name, imgs, annotations):
    id = [im['id'] for im in imgs if im['file_name'] == name]
    img_info = [im for im in imgs if im['file_name'] == name]
    img_annotation = [annot for annot in annotations if annot['image_id'] == id[0]]
    im_segmentations = [annot['segmentation'] for annot in annotations if annot['image_id'] == id[0]]
    im_bboxes = [annot['bbox'] for annot in annotations if annot['image_id'] == id[0]]
    return img_info[0], img_annotation, im_segmentations, im_bboxes


if __name__ == '__main__':
    file_name = '../params/0-ElBa_train.json'
    save_path = '../params/0-ElBa_selected_annotation.json'
    img_path = '../../doc/EIBa_selected_vecorization'

    names = img_names(img_path)
    parser = GetJsonInfo(file_name)
    imgs, annotations = parser.read_json()

    total_im_info = []
    total_im_annotations = []

    for name in names:
        im_info, im_annotation, im_segmentations, im_bboxes = get_img_info(name, imgs, annotations)
        total_im_info.append(im_info)
        for anno_inf in im_annotation:
            total_im_annotations.append(anno_inf)

    write_json(save_path, total_im_info, total_im_annotations)


    # #图片由信息中的coco_url获得
    # import requests
    # r = requests.get(img['coco_url'])
    # with open('./img.jpg', 'wb') as f:
    #     f.write(r.content)
    # image = cv.imread("./img.jpg", 1)
    # cv.namedWindow('IMG')
    # cv.imshow("IMG", image)
    # cv.waitKey()
    # cv.destroyAllWindows()
