# -*- coding:utf-8 -*-
import os
import time
import setting
import gc
from collections import defaultdict
import json
import pandas as pd
import shutil
from tqdm import trange
from collections import defaultdict
import lxml.etree as ET
from xml.dom import minidom

from DesignElementsExtraction import ForegroundElementsExtraction as FEE
from DesignElementsExtraction import RedundantElementsRemoval as RER
from DesignElementsVectorization import DesignElementVectorization as DEV
from EditableDesignGeneration import EditableDesignGeneration as EDG
from utils.data_process.record import record_excel, save_result_json, save_segments_json, write_json
from utils.data_process.calculate_svg_path import parse_svg, rename_save_svginfo

import warnings
warnings.filterwarnings("ignore")


class ImgVectorization(object):
    def __init__(self, files, visualization):
        self.files = files
        self.visualization = visualization

    def design_elements_extration(self):
        start_time = time.time()
        bk_color, cuttingImgs, seg_time, paths, bk_mask, de_segments, de_object_bboxes = FEE(self.files, start_time,
                                                                                             t=1, canny=False,
                                                                                             MBR=False,
                                                                                             initialtype='Felze',
                                                                                             visualization=self.visualization)
        element_extraction_time = round(time.time() - start_time, 1)

        keepPath, keep_elements = RER(self.files, cuttingImgs, paths, method="Perceptual Hashing",
                                      visualization=self.visualization)
        element_selection_time = round(time.time() - start_time - element_extraction_time, 1)
        num_elements = len(keep_elements)
        return paths, bk_color, num_elements, element_selection_time, element_extraction_time, start_time, [seg_time, bk_mask, de_segments, de_object_bboxes]

    def design_elements_vectorization(self, paths, bk_color, start_time, element_selection_time, element_extraction_time):
        SavePath, processPath, elementsPath, svgPath, keepPath = paths
        elements_info = []

        for ii in os.listdir(keepPath):
            element_info = DEV(ii, bk_color, self.files, paths, visualization=self.visualization)
            elements_info.append(element_info)
            gc.collect()

        theMoment = time.time()
        element_vectorization_time = round(theMoment - start_time - element_selection_time - element_extraction_time, 1)
        time_cost_all = round(theMoment - start_time, 1)
        return element_vectorization_time, time_cost_all, elements_info

    def design_generation(self):
        repeatType = ['straight', 'tile', 'half-drop']
        for t in repeatType:
            EDG(self.files, type=t)
        print(10 * '=' + 'over' + 10 * '=')

    def run(self, generate):
        result = []
        paths, bk_color, num_elements, element_selection_time, element_extraction_time, start_time, seg_result = self.design_elements_extration()
        element_vectorization_time, time_cost_all, elements_info = self.design_elements_vectorization(paths, bk_color, start_time, element_selection_time, element_extraction_time)
        if generate:
            self.design_generation()
        result.append([self.files, element_selection_time, element_extraction_time, element_vectorization_time, time_cost_all, num_elements])
        return result, elements_info, seg_result


def main(ImgPath):
    """keepPath在运行中不能修改，编辑，删除"""
    results = dict()
    seg_results = dict()
    files_all = []
    for files in os.listdir('./' + ImgPath):
        files_all.append(files)

    for k in trange(len(files_all), desc='Processing'):
        files = files_all[k]

        try:
            assert len(files.split(' ')) == 1   # 图片名不能有空格!
            print("image name: %s" % files)
            A = ImgVectorization(files, False)
            result, elements_info, seg_result = A.run(False)

            files_info_temp_dict = save_result_json(files, result, elements_info)
            temp_seg_results = save_segments_json(files, seg_result, result[0][2])
            results.setdefault('results', []).append(files_info_temp_dict)
            seg_results.setdefault('results', []).append(temp_seg_results)

        except Exception as E:
            print("Erro: " + files)
            print(E)
            with open(setting.SavePath + "ErroList.txt", "a") as f:
                f.write(str(files) + '\n')
                f.write(str(E) + '\n')
                f.write(' ' + '\n')
            if not os.path.exists('doc/Error/'):
                os.makedirs('doc/Error/')
            shutil.copy(os.path.join(setting.ImgPath, files), os.path.join('doc/Error/', files))
        continue
    time.sleep(1)
    return results, seg_results


if __name__ == '__main__':

    ImgPath = setting.ImgPath
    results, seg_results = main(ImgPath)
    overall_results = setting.SavePath + 'overall_results_6_json.json'
    write_json(data=results, savename=overall_results)
    write_json(data=seg_results, savename=setting.SavePath + 'seg_results_6_json.json')

    with open(overall_results, 'r') as load_f:
        results_ = json.load(load_f)
    record_excel(results_, setting.SavePath)

    """
    # rename, to revise a mistake
    paths = ['2-input-felz-output', '2-input-meanshift-output',
             '2-input-SLIC1000-output', '2-input-SLIC10000-output']
    for path in paths:
        curPath = 'doc/' + path + '/'
        info = rename_save_svginfo(curPath)
        df = pd.DataFrame(data=info)
        df.to_excel(os.path.join(curPath, 'overall_svg_path_info.xlsx'))
    
    #
    Path = 'doc/adobe-result/cc_low_feditly/'
    info = defaultdict(list)
    al = 0
    for svgName in os.listdir(Path):
        try:
            fileName = Path + svgName
            tree = ET.parse(open(fileName, 'r'))
            root = tree.getroot()
            doc = minidom.parse(fileName)  # parseString also exists
            path_strings = [path.getAttribute('d') for path
                            in doc.getElementsByTagName('path')]
            doc.unlink()

            num = len(path_strings)
            al += num
            info.setdefault(svgName, [num])
        except Exception as e:
            print(svgName)
            print(e)

    df = pd.DataFrame(data=info)
    df.to_excel(os.path.join('doc/output/', 'overall_adobe_svg_info.xlsx'))
    print(al / 667)
    """