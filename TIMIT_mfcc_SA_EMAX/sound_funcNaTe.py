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
    # orig_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.47574111208784
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 183.9878579984648
        minValue = -24.744228356823218
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_1_DR25_M_
    # """
    if PF_name == 'logEnergyFB':
        maxValue = 8.203314169657908
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -38.455734882115564
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    # """

    # orig_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.972418036245747
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 64.29384831191943
        minValue = -53.22628338839334
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.15169645729555
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 35.064132274052255
        minValue = -84.63177421775826
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """


    # pe_12_1_DR25_M_
    # 2_
    # """
    if PF_name == 'logEnergyFB':
        maxValue = 8.203314169657908
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -38.455734882115564
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    # """

    Nvalue = (value - minValue) / (maxValue - minValue)

    return Nvalue
