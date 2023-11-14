import os
import json
import numpy as np

ver = 'orig_50_M_librosa_'
ver = 'pe_50_M_librosa_'

ver = 'orig_50_librosa_'
ver = 'pe_50_librosa_'

ver += '2_'

now_path = r'D:\TIMITDIC_231101'
data_path = now_path + '_data_AIS'

ver_name_list = [ver]

for data_set in ['TEST', 'TRAIN']:

    # CNJF=CNN_Normalize_JsonFolder, CDF=CNN_Dataset_Folder, PF=ParmFolder, JF=json_file
    CNJF_path = os.path.join(data_path, data_set, 'cnn_normalize')
    CDF_path = os.path.join(data_path, data_set, 'cnn_dataset')

    PF_name_list = ['mfcc']

    type_name_list = ['F', 'M']

    if 'M' in ver:
        type_name_list = ['M']

    if 'F' in ver:
        type_name_list = ['F']

    for PF_name in PF_name_list:

        PF_path = os.path.join(CDF_path, PF_name)
        os.makedirs(PF_path, exist_ok=True)

        value = []
        label_type = []
        label_class = []

        for ver_name in ver_name_list:

            CNJF_PF_path = os.path.join(CNJF_path, ver_name + PF_name)

            for JF_obj in os.listdir(CNJF_PF_path):

                orig_JF_path = os.path.join(CNJF_PF_path, JF_obj)

                with open(orig_JF_path, 'r') as json_file:
                    orig_json_obj = json.load(json_file)

                for type_name in type_name_list:

                    for people in range(orig_json_obj[type_name]['people']):

                        value.append(orig_json_obj[type_name]['value'][people])
                        label_type.append(orig_json_obj[type_name]['typeNum'])
                        label_class.append(orig_json_obj['classNum'])

        if data_set == 'TEST':
            # 存入.json檔
            new_json_obj = {
                'TestLabel_class': label_class,
                'TestLabel_type': label_type,
                'TestValue': value,
            }

            JF_path = os.path.join(PF_path, ver + 'SoundDataset.json')

            with open(JF_path, 'w') as json_file:
                json.dump(new_json_obj, json_file, indent=4)

            # 存入.npz檔
            np.savez_compressed(
                os.path.join(PF_path, ver + 'SoundDataset.npz'),
                TestValue=np.array(value),
                TestLabel_type=np.array(label_type),
                TestLabel_class=np.array(label_class))
        else:
            # 打亂訓練集及測試集排序並微調分配
            index = np.random.permutation(np.arange(len(value)))
            rand_value = [value[i] for i in index]
            rand_label_type = [label_type[i] for i in index]
            rand_label_class = [label_class[i] for i in index]

            # 存入.json檔
            new_json_obj = {
                'TrainLabel_class': rand_label_class,
                'TrainLabel_type': rand_label_type,
                'TrainValue': rand_value,
            }

            JF_path = os.path.join(PF_path, ver + 'SoundDataset.json')

            with open(JF_path, 'w') as json_file:
                json.dump(new_json_obj, json_file, indent=4)

            # 存入.npz檔
            np.savez_compressed(
                os.path.join(PF_path, ver + 'SoundDataset.npz'),
                TrainValue=np.array(rand_value),
                TrainLabel_type=np.array(rand_label_type),
                TrainLabel_class=np.array(rand_label_class))


