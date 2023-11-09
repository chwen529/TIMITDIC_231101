import os
import json
from sound_funcMFCC import funcMFCC, PreEmphasis
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature

# sentence_type = 'SA1'
# sentence_type = 'SA2'
# sentence_type = 'SA'
sentence_type = 'SX'

ver = 'orig_50_M_'
ver = 'pe_50_M_'

ver += sentence_type + '_'

gen_img = False
# gen_img = True
gen_img_limit = 3

source_path = r'D:\TIMITDIC data'
now_path = r'D:\TIMITDIC_231101'
data_path = now_path + '_data_Italian_Spanish'

try:
    for data_set in ['TEST', 'TRAIN']:
    # for data_set in ['TRAIN']:
        print(data_set)

        SF_path = os.path.join(source_path, data_set)
        MJF_path = os.path.join(data_path, data_set, 'mfcc_json_librosa')

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

        # class_name_list = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
        class_name_list = ['DR2', 'CLIPS', 'soundboard']

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
                        data_path, data_set, 'img_librosa', ver + PF_name, class_name, type_name
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

                                if ('.WAV.wav' in CF_name_file and sentence_type in CF_name_file) or \
                                        (class_name == 'CLIPS') or (class_name == 'soundboard'):

                                    gen_img_count += 1

                                    AF_path = os.path.join(CF_path, CF_name, CF_name_file)
                                    AF_name, AF_ext = os.path.splitext(AF_path)

                                    if 'orig_50_' in ver:
                                        shape = 50

                                        y, sr = librosa.load(AF_path)
                                        f_MFCC_parm = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=shape)

                                    elif 'pe_50_' in ver:
                                        shape = 50

                                        y, sr = librosa.load(AF_path)

                                        PEsignal = PreEmphasis(y, 0.95)

                                        f_MFCC_parm = librosa.feature.mfcc(y=PEsignal, sr=sr, n_mfcc=shape)

                                    elif '_20_' in ver:
                                        shape = 20

                                        y, sr = librosa.load(AF_path)

                                        PEsignal = PreEmphasis(y, 0.95)

                                        # default n_mfcc = 20
                                        f_MFCC_parm = librosa.feature.mfcc(y=PEsignal, sr=sr)

                                    elif '_13_' in ver:
                                        shape = 13

                                        y, sr = librosa.load(AF_path)

                                        PEsignal = PreEmphasis(y, 0.95)

                                        f_MFCC_parm = librosa.feature.mfcc(y=PEsignal, sr=sr, n_mfcc=shape)

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

                                        plt.savefig(
                                            os.path.join(
                                                img_path,
                                                CF_name + '_' + CF_name_file.replace(
                                                    '.WAV.wav' if class_name != 'CLIPS' else '.wav', '.png'
                                                )
                                            ),
                                            bbox_inches='tight',
                                            pad_inches=0
                                        )

                                        plt.clf()
                                        ax = plt.subplot()

                                        # 二維
                                        if PF_name not in ['signal', 'PEsignal']:
                                            ax.imshow(f_MFCC_parm.tolist(), cmap='hsv')

                                            plt.savefig(
                                                os.path.join(
                                                    img_2d_path,
                                                    CF_name + '_' + CF_name_file.replace(
                                                        '.WAV.wav' if class_name != 'CLIPS' else '.wav', '_2d.png'
                                                    )
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
