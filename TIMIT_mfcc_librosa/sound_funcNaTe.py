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
        # maxValue = 241.40103149414062
        # minValue = -760.611572265625
        # ##############################################
        # pe_50_DR25_M_
        # maxValue = 174.31935119628906
        # minValue = -851.4518432617188
        # ##############################################
        # pe_50_M_
        # maxValue = 176.81271362304688
        # minValue = -856.1064453125
        # ##############################################
        # pe_20_DR25_M_
        # maxValue = 168.72964477539062
        # minValue = -851.4518432617188
        # ##############################################
        # pe_13_DR25_M_
        maxValue = 168.72964477539062
        minValue = -812.615966796875
        # ##############################################
        # ##############################################
        # ##############################################
        # 2_
        # orig_50_DR25_M_
        # maxValue = 243.99072265625
        # minValue = -799.1029663085938
        # ##############################################
        # pe_50_DR25_M_
        # maxValue = 179.03524780273438
        # minValue = -897.0848388671875
        # ##############################################
        # pe_50_M_
        # maxValue = 194.22293090820312
        # minValue = -897.0848388671875
        # ##############################################
        # pe_20_DR25_M_
        # maxValue = 178.67974853515625
        # minValue = -897.0848388671875
        # ##############################################
        # pe_13_DR25_M_
        # maxValue = 176.87413024902344
        # minValue = -897.0848388671875
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    # """
    # ########################################################################################
    # SX
    """
    if PF_name == 'mfcc':
        # orig_50_DR25_M_
        # maxValue = 245.124755859375
        # minValue = -776.5081787109375
        # ##############################################
        # pe_50_DR25_M_
        # maxValue = 179.22454833984375
        # minValue = -869.7867431640625
        # ##############################################
        # ##############################################
        # ##############################################
        # 2_
        # orig_50_DR25_M_
        # maxValue = 247.6973419189453
        # minValue = -808.1175537109375
        # ##############################################
        # pe_50_DR25_M_
        maxValue = 177.99197387695312
        minValue = -903.5643920898438
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    Nvalue = (value - minValue) / (maxValue - minValue)

    NvalueMaxValue = np.max(Nvalue)
    NvalueMinValue = np.min(Nvalue)

    if NvalueMaxValue > 1:
        print(NvalueMaxValue)

    if NvalueMinValue < 0:
        print(NvalueMinValue)

    return Nvalue
