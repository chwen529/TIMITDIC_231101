import os
import json
from sound_funcMFCC import funcMFCC
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy

sentence_type = 'SA1'
# sentence_type = 'SA2'

condition = ['DR25', 'M']

ver_list = [
    # 'orig_50_', 'pe_50_', 'butter_50_'
    'orig_30_', 'pe_30_', 'butter_30_'
    'orig_12_', 'pe_12_', 'butter_12_'
]

gen_img = False
gen_img = True
gen_img_limit = 3

source_path = r'D:\TIMITDIC data'
data_path = r'D:\TIMITDIC_analyze_different'

try:
    # for data_set in ['TEST', 'TRAIN']:
    for data_set in ['TRAIN']:
        print(data_set)

        SF_path = os.path.join(source_path, data_set)
        MJF_path = os.path.join(data_path, data_set, 'mfcc_json')

        PF_name_list = [
            'signal',
            'PEsignal',
            'frames',
            'magSpectr',
            'logMagSpectr',
            'powSpectr',
            'logPowSpectr',
            'filterBanks',
            'logEnergyFB',
            'mfcc'
        ]

        class_name_list = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
        type_name_list = ['female', 'male']

        if 'DR25' in condition:
            class_name_list = ['DR2', 'DR5']

        if 'male' in condition:
            type_name_list = ['male']

        if 'female' in condition:
            type_name_list = ['female']

        # Create MFCC_JsonFile
        for PF_name in PF_name_list:

            for ver in ver_list:

                # Create MFCC_JsonFolder_ParmFolder
                PF_path = os.path.join(MJF_path, PF_name)
                os.makedirs(PF_path, exist_ok=True)

                for class_name in class_name_list:

                    CF_path = os.path.join(SF_path, class_name)

                    json_obj = {
                        'title': 'mfcc_json',
                        'parm': ver + PF_name,
                        'class': class_name,
                        'classNum': class_name_list.index(class_name),
                        'rate': 16000
                    }

                    for type_name in type_name_list:

                        img_path = os.path.join(
                            data_path, data_set, 'img', PF_name, type_name
                        )
                        img_2d_path = os.path.join(img_path, '2d')

                        if gen_img:
                            os.makedirs(img_path, exist_ok=True)

                            if PF_name not in ['signal', 'PEsignal']:
                                os.makedirs(img_2d_path, exist_ok=True)

                        id_list = []
                        shape_list = []
                        value_list = []

                        gen_img_count = 0

                        # 遍歷單一區檔案資料夾，此處資料夾為單一人，ex.FCJF0
                        for CF_name in os.listdir(CF_path):

                            # 性別判定
                            if CF_name[0].lower() == type_name[0].lower():

                                # 遍歷單一人檔案，ex.SA1.WAV.wav
                                for CF_name_file in os.listdir(os.path.join(CF_path, CF_name)):

                                    if '.WAV.wav' in CF_name_file and sentence_type in CF_name_file:

                                        if gen_img_count >= gen_img_limit:
                                            break

                                        gen_img_count += 1

                                        AF_path = os.path.join(CF_path, CF_name, CF_name_file)
                                        AF_name, AF_ext = os.path.splitext(AF_path)

                                        if 'orig_12_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=26, start_nCeps=1, end_nCeps=13
                                            )

                                        elif 'pe_12_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=26, start_nCeps=1, end_nCeps=13,
                                                preEmphasis=True, preCoeff=0.95
                                            )

                                        elif 'butter_12_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=26, start_nCeps=1, end_nCeps=13,
                                                butter=True
                                            )

                                        elif 'orig_30_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=30, start_nCeps=0, end_nCeps=30
                                            )

                                        elif 'pe_30_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=30, start_nCeps=0, end_nCeps=30,
                                                preEmphasis=True, preCoeff=0.95
                                            )

                                        elif 'butter_30_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=30, start_nCeps=0, end_nCeps=30,
                                                butter=True
                                            )

                                        elif 'orig_50_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=50, start_nCeps=0, end_nCeps=50
                                            )

                                        elif 'pe_50_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=50, start_nCeps=0, end_nCeps=50,
                                                preEmphasis=True, preCoeff=0.95
                                            )

                                        elif 'butter_50_' in ver:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=50, start_nCeps=0, end_nCeps=50,
                                                butter=True
                                            )

                                        else:
                                            continue

                                        id_list.append(ver + CF_name + '_' + CF_name_file.split('.')[0])
                                        shape_list.append(f_MFCC_parm.shape)
                                        value_list.append(f_MFCC_parm.tolist())

                                        if gen_img and gen_img_count <= gen_img_limit:
                                            # ########## 生成圖片 ##########
                                            plt.clf()
                                            ax = plt.subplot()

                                            # 一維
                                            if PF_name in ['signal', 'PEsignal']:
                                                ax.plot(f_MFCC_parm.tolist())
                                            else:
                                                ax.plot(f_MFCC_parm.flatten())

                                            plt.savefig(
                                                os.path.join(
                                                    img_path,
                                                    ver + class_name + '_' + CF_name + '_' +
                                                    CF_name_file.replace('.WAV.wav', '.png')
                                                ),
                                                bbox_inches='tight',
                                                pad_inches=0
                                            )

                                            plt.clf()
                                            ax = plt.subplot()

                                            # 二維
                                            if PF_name not in ['signal', 'PEsignal']:
                                                ax.imshow(f_MFCC_parm.tolist(), cmap='hsv')

                                                # title = data['class'] + '_' + cate2
                                                # ax.set_title(title, fontsize=10, c='red')
                                                # ax.set_xticks([])
                                                # ax.set_yticks([])  # 設定不顯示刻度
                                                # ax.axis('off')

                                                plt.savefig(
                                                    os.path.join(
                                                        img_2d_path,
                                                        ver + class_name + '_' + CF_name + '_' +
                                                        CF_name_file.replace('.WAV.wav', '_2d.png')
                                                    ),
                                                    bbox_inches='tight',
                                                    pad_inches=0
                                                )

                        json_obj[type_name] = {
                            'type': type_name,
                            'typeNum': type_name_list.index(type_name),
                            'people': len(id_list),
                            'id': id_list,
                            'shape': shape_list,
                            'value': value_list
                        }

                    JF_path = os.path.join(PF_path, ver + PF_name + '_' + class_name + '.json')

                    with open(JF_path, 'w') as json_file:
                        json.dump(json_obj, json_file, indent=4)

except Exception as e:
    import traceback
    print(traceback.format_exc())
    print(str(e))
