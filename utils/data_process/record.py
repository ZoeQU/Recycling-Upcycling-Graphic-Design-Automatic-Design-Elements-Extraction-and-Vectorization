# -*- coding:utf-8 -*-
import pandas as pd
import json
import csv
import os


def save_res_csv(csv_head, res_name, data):
    # # recording
    # with open('doc/output_EIBa_test/records.txt', 'w') as f:
    #     f.write(str(x) + '\n')
    # f.close()

    with open(res_name, 'w', newline='') as f:  #后面补上newline=''可以去掉空白行
        csv_write = csv.writer(f)
        # csv_head = ['image_name', 'dx', 'dy', 'precision', 'propensity', 'time_cost']
        csv_write.writerow(csv_head)
        for i in data:
            csv_write.writerow(i)


def record_excel(data, filepath):
    '''按列写入'''
    data2 = data['results']
    columns = []
    for dd in data2:
        column = []
        for key, value in dd.items():
            column.append(dd[key])
        columns.append([column[i] for i in range(len(column) - 1)])
    titles = [key for key, value in data2[0].items()]
    new_data1 = []
    for i in range(len(data2)):
        temp_info_dict = {}
        for j in range(len(titles) - 1):
            temp_info_dict[titles[j]] = columns[i][j]
        new_data1.append(temp_info_dict)
    df1 = pd.DataFrame(data=new_data1)

    design_elements_info = [dd.get('design_elements_info') for dd in data2]
    new_data2 = []
    for des in design_elements_info:
        for i, col in enumerate(des):
            new_data2.append(col)
    df2 = pd.DataFrame(data=new_data2)

    with pd.ExcelWriter(os.path.join(filepath, 'overall_results.xlsx'), mode='w', engine='openpyxl') as writer:
        df1.to_excel(writer, index=False, sheet_name='overall')
        df2.to_excel(writer, index=False, sheet_name='vectorization_info')
    print('=' * 10 + 'saved' + '=' * 10)


def save_result_json(files, result, elements_info):
    files_info_temp_dict = dict()
    files_info_temp_dict['image_name'] = files
    files_info_temp_dict['element_selection_time'] = result[0][1]
    files_info_temp_dict['element_extraction_time'] = result[0][2]
    files_info_temp_dict['element_vectorization_time'] = result[0][3]
    files_info_temp_dict['time_cost_all'] = result[0][4]
    files_info_temp_dict['num_elements'] = result[0][5]
    files_info_temp_dict['design_elements_info'] = []
    for de_info in elements_info:
        de_info_temp_dict = dict()
        # [im_size, filename_ori, num_ori, filename_filter, num_af_filter, timecost]
        de_info_temp_dict['design_element_size'] = de_info[0]
        de_info_temp_dict['design_element_name_ori'] = de_info[1]
        de_info_temp_dict['num_path_ori'] = de_info[2]
        de_info_temp_dict['design_element_name_af_filter'] = de_info[3]
        de_info_temp_dict['num_path_af_filter'] = de_info[4]
        de_info_temp_dict['vectorization_time'] = de_info[5]
        files_info_temp_dict['design_elements_info'].append(de_info_temp_dict)
    return files_info_temp_dict


def save_segments_json(files, save_info, de_extraction_time):
    seg_time, bk_mask, de_segments, de_object_bboxes = save_info
    temp_results = dict()
    temp_results['image_name'] = files
    temp_results['seg_time'] = seg_time
    temp_results['design_elements_extraction_time'] = de_extraction_time
    temp_results['bk_mask'] = bk_mask.tolist()
    temp_results['segments'] = []
    for i, segs in enumerate(de_segments):
        temp_results['segments'].append({'id': i, 'coords': segs.tolist()})
    temp_results['bbox'] = []
    for j, de_bbox in enumerate(de_object_bboxes):
        temp_results['bbox'].append({'id': j, 'coords': de_bbox})
    return temp_results


def write_json(data, savename):
    b = json.dumps(data)
    f2 = open(savename, 'w')
    f2.write(b)
    f2.close()

