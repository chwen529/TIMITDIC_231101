import numpy as np


def getMidRange(valueLen, needLen=1):
    lenStart = int(valueLen / 2 - (needLen - 1) / 2)
    lenEnd = int(valueLen / 2 + (needLen + 1) / 2)

    return lenStart, lenEnd


def getMidValue(value, needLen, outVer=None):

    if outVer == '2_':
        # 在第一维和第二维上进行等距采样
        sampled_values_1st_dim = np.linspace(0, value.shape[0] - 1, needLen, dtype=int)
        sampled_values_2nd_dim = np.linspace(0, value.shape[1] - 1, needLen, dtype=int)

        # 使用采样的索引获取子数组
        sampled_array = value[sampled_values_1st_dim][:, sampled_values_2nd_dim]

        return sampled_array

    lenStart = int(len(value) / 2 - (needLen - 1) / 2)
    lenEnd = int(len(value) / 2 + (needLen + 1) / 2)

    lenStart2 = int(len(value[0]) / 2 - (needLen - 1) / 2)
    lenEnd2 = int(len(value[0]) / 2 + (needLen + 1) / 2)

    return value[lenStart:lenEnd, lenStart2:lenEnd2]


def getNormalizeValue(value):
    import numpy as np
    maxValue = np.max(value)
    minValue = np.min(value)
    Nvalue = (value - minValue) / (maxValue - minValue)

    # avgValue = np.average(value)
    # Nvalue = value / (avgValue * 1.5)
    #
    # for i in range(len(Nvalue)):
    #     for j in range(len(Nvalue[i])):
    #         if Nvalue[i][j] > 1:
    #             Nvalue[i][j] = 1

    return Nvalue


def getNormalizeValueNew(value, PF_name):
    import numpy as np

    # SA1
    # """
    if PF_name == 'mfcc':
        # orig_50_DR25_M_
        # maxValue = 242.4998779296875
        # minValue = -763.6142578125
        # ##############################################
        # pe_50_M_
        # maxValue = 173.07748413085938
        # minValue = -856.7965698242188
        # ##############################################
        # ##############################################
        # ##############################################
        # 2_
        # orig_50_DR25_M_
        # maxValue = 243.26878356933594
        # minValue = -808.1175537109375
        # ##############################################
        # pe_50_DR25_M_
        maxValue = 166.48165893554688
        minValue = -891.3726196289062
        # pe_50_librosa_
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    # """

    Nvalue = (value - minValue) / (maxValue - minValue)

    NvalueMaxValue = np.max(Nvalue)
    NvalueMinValue = np.min(Nvalue)

    if NvalueMaxValue > 1:
        print(NvalueMaxValue)

    if NvalueMinValue < 0:
        print(NvalueMinValue)

    return Nvalue
