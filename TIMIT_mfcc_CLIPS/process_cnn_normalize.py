import os, json
import numpy as np
from sound_funcNaTe import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import openpyxl

sentence_type = 'SA1'
# sentence_type = 'SA2'
# sentence_type = 'SA'
# sentence_type = 'SX'

ver = 'orig_50_DR25_M_'
ver = 'pe_50_DR25_M_'

ver += sentence_type + '_'

outVer = ''
outVer = '2_'

gen_img = False
gen_img = True
gen_img_limit = 3

now_path = r'D:\TIMITDIC_231101'
data_path = now_path + '_data_CLIPS'

try:
    for data_set in ['TEST', 'TRAIN']:
    # for data_set in ['TRAIN']:
        print(data_set)

        MJF_path = os.path.join(data_path, data_set, 'mfcc_json_librosa')
        CNJF_path = os.path.join(data_path, data_set, 'cnn_normalize_librosa')

        PF_name_list = ['mfcc']

        type_name_list = ['F', 'M']

        if 'M' in ver:
            type_name_list = ['M']

        if 'F' in ver:
            type_name_list = ['F']

        # MFCC_JsonFile to CNN_JsonFile
        for PF_name in PF_name_list:

            # 單一參數_最大最小值
            PF_max_value = 0
            PF_min_value = 0

            # Create CNN_JsonFolder_ParmFolder
            PF_path = os.path.join(CNJF_path, ver + outVer + PF_name)
            os.makedirs(PF_path, exist_ok=True)

            MJF_PF_path = os.path.join(MJF_path, ver + PF_name)

            for JF_obj in os.listdir(MJF_PF_path):

                orig_JF_path = os.path.join(MJF_PF_path, JF_obj)

                with open(orig_JF_path, 'r') as json_file:
                    orig_json_obj = json.load(json_file)

                new_json_obj = orig_json_obj
                new_json_obj['title'] = 'cnn_normalize'

                for type_name in type_name_list:

                    img_path = os.path.join(
                        data_path, data_set, 'img_librosa', ver + outVer + 'cnn_' + PF_name, new_json_obj['class'], type_name
                    )
                    img_2d_path = os.path.join(img_path, '2d')
                    img_value_path = os.path.join(img_path, 'value')
                    img_mid_value_path = os.path.join(img_path, 'midValue')

                    if gen_img:
                        os.makedirs(img_path, exist_ok=True)
                        os.makedirs(img_2d_path, exist_ok=True)
                        os.makedirs(img_value_path, exist_ok=True)
                        os.makedirs(img_mid_value_path, exist_ok=True)

                    shape_list = []
                    value_list = []

                    gen_img_count = 0

                    for people in range(orig_json_obj[type_name]['people']):

                        value = np.array(orig_json_obj[type_name]['value'][people])

                        if 'orig_50_' in ver or 'pe_50_' in ver:
                            needLen = 50

                            midValue = getMidValue(value, needLen, outVer)

                        elif '_20_' in ver:
                            needLen = 20

                            midValue = getMidValue(value, needLen, outVer)

                        elif '_13_' in ver:
                            needLen = 13

                            midValue = getMidValue(value, needLen, outVer)

                        else:
                            needLen = 50

                            midValue = getMidValue(value, needLen, outVer)

                        if np.max(midValue) > PF_max_value:
                            PF_max_value = np.max(midValue)

                        if np.min(midValue) < PF_min_value:
                            PF_min_value = np.min(midValue)

                        Nvalue = getNormalizeValueNew(midValue, PF_name)
                        shape_list.append(Nvalue.shape)
                        value_list.append(Nvalue.tolist())

                        if gen_img and gen_img_count < gen_img_limit:
                            gen_img_count += 1
                            # ######### 生成圖片 ##########
                            plt.clf()
                            ax = plt.subplot()
                            plt.ylim(0, 1)

                            # 一維 ### Nvalue ###
                            ax.plot(Nvalue.flatten())

                            plt.savefig(
                                os.path.join(
                                    img_path,
                                    orig_json_obj[type_name]['id'][people].replace(ver, '')
                                ),
                                bbox_inches='tight',
                                pad_inches=0
                            )

                            plt.clf()
                            ax = plt.subplot()

                            # 一維 ### value ###
                            ax.plot(value.flatten())

                            plt.savefig(
                                os.path.join(
                                    img_value_path,
                                    orig_json_obj[type_name]['id'][people].replace(ver, '') + '_value'
                                ),
                                bbox_inches='tight',
                                pad_inches=0
                            )

                            plt.clf()
                            ax = plt.subplot()

                            # 一維 ### midValue ###
                            ax.plot(midValue.flatten())

                            plt.savefig(
                                os.path.join(
                                    img_mid_value_path,
                                    orig_json_obj[type_name]['id'][people].replace(ver, '') + '_midValue'
                                ),
                                bbox_inches='tight',
                                pad_inches=0
                            )

                            # 一維拆解
                            for cate in ['value', 'midValue', 'Nvalue']:
                                img_split_save_path = os.path.join(
                                    img_path,
                                    orig_json_obj[type_name]['id'][people].replace(ver, ''),
                                    cate
                                )
                                os.makedirs(img_split_save_path, exist_ok=True)

                                for i in range(len(Nvalue)):
                                    plt.clf()
                                    ax = plt.subplot()

                                    if cate == 'value':
                                        ax.plot(value[int(len(value) / 2 - (needLen - 1) / 2) + i])
                                    elif cate == 'midValue':
                                        ax.plot(midValue[i])
                                    elif cate == 'Nvalue':
                                        ax.plot(Nvalue[i])

                                    plt.savefig(
                                        os.path.join(
                                            img_split_save_path,
                                            orig_json_obj[type_name]['id'][people].replace(ver, '') +
                                            '_split_' + cate + '_' + str(i)
                                        ),
                                        bbox_inches='tight',
                                        pad_inches=0
                                    )

                            plt.clf()
                            ax = plt.subplot()

                            # 二維
                            ax.imshow(Nvalue.tolist(), cmap='hsv')

                            plt.savefig(
                                os.path.join(
                                    img_2d_path,
                                    orig_json_obj[type_name]['id'][people].replace(ver, '') + '_2d'
                                ),
                                bbox_inches='tight',
                                pad_inches=0
                            )

                    new_json_obj[type_name]['shape'] = shape_list
                    new_json_obj[type_name]['value'] = value_list

                new_JF_path = os.path.join(PF_path, JF_obj)

                with open(new_JF_path, 'w') as json_file:
                    json.dump(new_json_obj, json_file, indent=4)

            print(PF_name)
            print('max_value:' + str(PF_max_value))
            print('min_value:' + str(PF_min_value))

except Exception as e:
    import traceback
    print(traceback.format_exc())
    print(str(e))
