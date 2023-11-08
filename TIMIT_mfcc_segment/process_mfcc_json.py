import os
import json
from sound_funcMFCC import funcMFCC, PreEmphasis
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
from pydub import AudioSegment
from pydub.silence import split_on_silence

sentence_type = 'SA1'
# sentence_type = 'SA2'
# sentence_type = 'SA'
# sentence_type = 'SX'

ver = 'pe_3_DR25_M_'
# --------------------------------------------------

ver += sentence_type + '_'

gen_img = False
# gen_img = True
gen_img_limit = 1

source_path = r'D:\TIMITDIC data_split_matlab_SA1_DR25_M\threshold0.03_frameLen100_silentMax300'
now_path = r'D:\TIMITDIC_231101'
data_path = now_path + '_data_segment'

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

        class_name_list = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']

        if 'DR25' in ver:
            class_name_list = ['DR2', 'DR5']

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

                json_obj = {
                    'title': 'mfcc_json',
                    'parm': ver + PF_name,
                    'class': class_name,
                    'classNum': class_name_list.index(class_name),
                    'rate': 16000
                }

                for type_name in type_name_list:

                    CF_path = os.path.join(SF_path, type_name, class_name)

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

                            # 單一人檔案只取10個
                            CF_name_file_count = 0
                            CF_name_file_limit = 5
                            # 存放這10個音訊的特徵
                            f_MFCC_parm_list = np.array([])

                            # 遍歷單一人檔案，ex.SA1.WAV.wav
                            for CF_name_file in os.listdir(os.path.join(CF_path, CF_name)):

                                if '.wav' in CF_name_file and sentence_type in CF_name_file:

                                    if 'max' in CF_name_file or CF_name_file_count > CF_name_file_limit:
                                        continue

                                    AF_path = os.path.join(CF_path, CF_name, CF_name_file)
                                    AF_name, AF_ext = os.path.splitext(AF_path)

                                    if '_20_' in ver:
                                        shape = 20

                                        y, sr = librosa.load(AF_path)

                                        PEsignal = PreEmphasis(y, 0.95)

                                        # default n_mfcc = 20
                                        f_MFCC_parm = librosa.feature.mfcc(y=PEsignal, sr=sr)

                                    elif '_3_' in ver:
                                        shape = 3

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

                                    CF_name_file_count += 1
                                    gen_img_count += 1

                                    if f_MFCC_parm_list.shape[0] == 0:
                                        f_MFCC_parm_list = f_MFCC_parm
                                    else:
                                        f_MFCC_parm_list = np.vstack((f_MFCC_parm_list, f_MFCC_parm))

                            if len(f_MFCC_parm_list) < CF_name_file_limit:
                                continue

                            f_MFCC_parm_list = np.array(f_MFCC_parm_list)

                            id_list.append(ver + CF_name + '_' + 'SA1')
                            shape_list.append(f_MFCC_parm_list.shape)
                            value_list.append(f_MFCC_parm_list.tolist())

                            if gen_img and gen_img_count <= gen_img_limit:
                                # ########## 生成圖片 ##########
                                plt.clf()
                                ax = plt.subplot()

                                # 一維
                                if PF_name in ['signal', 'PEsignal']:
                                    ax.plot(f_MFCC_parm_list.tolist())
                                else:
                                    ax.plot(f_MFCC_parm_list.flatten())

                                plt.savefig(
                                    os.path.join(
                                        img_path,
                                        CF_name + '_SA1.png'
                                    ),
                                    bbox_inches='tight',
                                    pad_inches=0
                                )

                                plt.clf()
                                ax = plt.subplot()

                                # 二維
                                if PF_name not in ['signal', 'PEsignal']:
                                    ax.imshow(f_MFCC_parm_list.tolist(), cmap='hsv')

                                    plt.savefig(
                                        os.path.join(
                                            img_2d_path,
                                            CF_name + '_SA1_2d.png'
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
