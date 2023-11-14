import os
import json
from sound_funcMFCC import funcMFCC, PreEmphasis
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature

# ver = 'orig_50_M_'
# ver = 'pe_50_M_'

ver = 'orig_50_M_librosa_'
ver = 'pe_50_M_librosa_'

ver = 'orig_50_librosa_'
ver = 'pe_50_librosa_'

gen_img = False
gen_img = True
gen_img_limit = 3

source_path = r'D:\TIMITDIC data'
now_path = r'D:\TIMITDIC_231101'
data_path = now_path + '_data_AIS'

try:
    for data_set in ['TEST', 'TRAIN']:
    # for data_set in ['TRAIN']:
        print(data_set)

        SF_path = os.path.join(source_path, data_set)
        MJF_path = os.path.join(data_path, data_set, 'mfcc_json')

        PF_name_list = [
            # 'signal',
            # 'PEsignal',
            # 'frames',
            # 'magSpectr',
            # 'logMagSpectr',
            # 'powSpectr',
            # 'logPowSpectr',
            # 'filterBanks',
            # 'logEnergyFB',
            'mfcc'
        ]

        class_name_list = ['TIMIT', 'CLIPS', 'soundboard']

        type_name_list = ['F', 'M']

        if 'M' in ver:
            type_name_list = ['M']

        if 'F' in ver:
            type_name_list = ['F']

        # Create MFCC_JsonFile
        for PF_name in PF_name_list:

            # Create MFCC_JsonFolder_ParmFolder
            PF_path = os.path.join(MJF_path, ver + PF_name)
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
                        data_path, data_set, 'img', ver + PF_name, class_name, type_name
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

                                if ('.WAV.wav' in CF_name_file and class_name == 'TIMIT') or \
                                        ('.wav' in CF_name_file and class_name == 'CLIPS') or \
                                        ('.mp3' in CF_name_file and class_name == 'soundboard'):

                                    gen_img_count += 1

                                    AF_path = os.path.join(CF_path, CF_name, CF_name_file)
                                    AF_name, AF_ext = os.path.splitext(AF_path)

                                    if 'orig_50_' in ver:
                                        shape = 50

                                        if 'librosa' in ver:
                                            y, sr = librosa.load(AF_path)
                                            f_MFCC_parm = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=shape)
                                        else:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=shape, start_nCeps=0, end_nCeps=shape
                                            )

                                    elif 'pe_50_' in ver:
                                        shape = 50

                                        if 'librosa' in ver:
                                            y, sr = librosa.load(AF_path)

                                            PEsignal = PreEmphasis(y, 0.95)

                                            f_MFCC_parm = librosa.feature.mfcc(y=PEsignal, sr=sr, n_mfcc=shape)
                                        else:
                                            f_MFCC_parm = funcMFCC(
                                                AF_path, PF_name, nBanks=shape, start_nCeps=0, end_nCeps=shape,
                                                preEmphasis=True, preCoeff=0.95
                                            )

                                    else:
                                        shape = 50

                                        y, sr = librosa.load(AF_path)
                                        f_MFCC_parm = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=shape)

                                    if len(f_MFCC_parm) < shape or len(f_MFCC_parm[0]) < shape:
                                        print(CF_name_file.split('.')[0])
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

                                        tmp_CF_name_file = CF_name_file

                                        if class_name == 'TIMIT':
                                            tmp_CF_name_file = tmp_CF_name_file.replace('.WAV.wav', '.png')
                                        elif class_name == 'CLIPS':
                                            tmp_CF_name_file = tmp_CF_name_file.replace('.wav', '.png')
                                        elif class_name == 'soundboard':
                                            tmp_CF_name_file = tmp_CF_name_file.replace('.mp3', '.png')
                                        else:
                                            tmp_CF_name_file += '.png'

                                        plt.savefig(
                                            os.path.join(
                                                img_path,
                                                CF_name + '_' + tmp_CF_name_file
                                            ),
                                            bbox_inches='tight',
                                            pad_inches=0
                                        )

                                        plt.clf()
                                        ax = plt.subplot()

                                        # 二維
                                        if PF_name not in ['signal', 'PEsignal']:
                                            # 二維
                                            im = ax.imshow(f_MFCC_parm.tolist(), cmap='hsv')

                                            # Add colorbar
                                            cbar = plt.colorbar(im, ax=ax)

                                            plt.savefig(
                                                os.path.join(
                                                    img_2d_path,
                                                    CF_name + '_2d_' + tmp_CF_name_file
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
