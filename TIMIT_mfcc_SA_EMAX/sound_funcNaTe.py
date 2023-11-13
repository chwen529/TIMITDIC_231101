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
    # """
    if PF_name == 'logEnergyFB':
        maxValue = 9.47574111208784
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 183.9878579984648
        minValue = -24.744228356823218
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    # """

    # pe_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.203314169657908
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -38.455734882115564
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

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
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # orig_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.47574111208784
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 183.9878579984648
        minValue = -27.345572788660732
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.217726325944714
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -38.455734882115564
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.972418036245747
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 66.03973734216129
        minValue = -60.33618198851738
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.160808881110416
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 37.38743922596362
        minValue = -84.63177421775826
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # SA2
    # orig_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.227664412227398
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 179.64435088943037
        minValue = -19.465031075028826
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.309794639297948
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 167.5923719668694
        minValue = -32.14613421845251
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.88607760719294
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 58.55590209176884
        minValue = -43.00823276475066
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.055690164319476
        minValue = -0.017811774115503925
    elif PF_name == 'mfcc':
        maxValue = 28.516887652088997
        minValue = -70.23351574077182
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # orig_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.227664412227398
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 179.64435088943037
        minValue = -19.465031075028826
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.309794639297948
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 167.5923719668694
        minValue = -32.14613421845251
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.88607760719294
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 68.0298860048652
        minValue = -43.00823276475066
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.055690164319476
        minValue = -0.017811774115503925
    elif PF_name == 'mfcc':
        maxValue = 39.079340906665664
        minValue = -70.23351574077182
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # 2_
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # SA1
    # orig_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.47574111208784
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 183.92628129657425
        minValue = -24.74422835682322
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.203314169657908
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -38.455734882115564
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.40000740349855
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
        maxValue = 8.176345242724217
        minValue = -0.8099393523517843
    elif PF_name == 'mfcc':
        maxValue = 35.064132274052255
        minValue = -84.63177421775826
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # orig_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.47574111208784
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 183.9878579984648
        minValue = -27.345572788660732
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.217726325944714
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 171.5050046672282
        minValue = -39.32222294622575
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.40000740349855
        minValue = -0.02500556628082717
    elif PF_name == 'mfcc':
        maxValue = 66.03973734216129
        minValue = -60.33618198851738
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.176345242724217
        minValue = -1.2779512074308876
    elif PF_name == 'mfcc':
        maxValue = 36.88728573137601
        minValue = -87.97073649772435
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # SA2
    # orig_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.227664412227398
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 179.64435088943037
        minValue = -27.64057371635772
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.309794639297948
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 167.5923719668694
        minValue = -39.66442740776446
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.842841061085439
        minValue = -0.022774466478436728
    elif PF_name == 'mfcc':
        maxValue = 65.6674495786134
        minValue = -62.214718749511874
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_1_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.226776451471993
        minValue = -1.203187034779999
    elif PF_name == 'mfcc':
        maxValue = 35.30560951086633
        minValue = -87.7521799669282
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """
    # ##############################################################
    # ##############################################################
    # ##############################################################
    # orig_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 9.227664412227398
        minValue = 0
    elif PF_name == 'mfcc':
        maxValue = 179.64435088943037
        minValue = -27.64057371635772
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_12_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.309794639297948
        minValue = -0.34863202044626784
    elif PF_name == 'mfcc':
        maxValue = 167.5923719668694
        minValue = -39.66442740776446
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # orig_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.8521879719711
        minValue = -0.022774466478436728
    elif PF_name == 'mfcc':
        maxValue = 69.11730937372323
        minValue = -62.214718749511874
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    # pe_2612_3_DR25_M_
    """
    if PF_name == 'logEnergyFB':
        maxValue = 8.226776451471993
        minValue = -1.3255455491304793
    elif PF_name == 'mfcc':
        maxValue = 36.594649456495276
        minValue = -87.7521799669282
    else:
        maxValue = np.max(value)
        minValue = np.min(value)
    """

    Nvalue = (value - minValue) / (maxValue - minValue)

    return Nvalue
