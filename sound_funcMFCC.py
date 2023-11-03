import numpy as np
import pywt
from scipy.io.wavfile import read
from scipy.fft import dct
import matplotlib.pyplot as plt
from scipy import signal


def PreEmphasis(signal, preCoeff=0.97):
    # coeff: 0.95 or 0.97, std is 0.97
    # -------------------------------------------------- #
    # 音訊預強調: 高通濾波器
    # 輸入: 音訊,(預強調係數)
    # 輸出: 預強調音訊, shape:(音訊數)
    PEsignal = np.append(signal[0], signal[1:] - preCoeff * signal[:-1])

    return PEsignal


def Framing(PEsignal, rate, frameTime=0.025, overlap=0.6):
    # frameSize: 20 ~ 40ms, std is 25ms
    # overlap: 50%(+/-10)% or 1/3, std is none
    # -------------------------------------------------- #
    # 音訊分幀: 每25ms分一幀，每幀重疊率60%
    # 輸入: (預加重)音訊,採樣率,(每幀時長,重疊率)
    # 輸出: 分幀音訊, shape:(分幀數,每幀採樣數)
    signalLen = len(PEsignal)
    frameLen = int(rate * frameTime)
    frameStep = int(frameLen * (1 - overlap))
    frames = [PEsignal[i:i + frameLen] for i in np.arange(0, signalLen - frameLen, frameStep)]
    frames = np.array(frames)

    return frames


def FourierTransform(frames):
    # 傅立葉轉換: 每幀進行實數快速傅立葉變換(rFFT)
    # 輸入: 分幀音訊
    # 輸出: 幅級譜, shape:(分幀數,每幀採樣數/2+1)
    nFFT = len(frames[0])
    window = np.hamming(nFFT)
    magSpectr = np.abs(np.fft.rfft(frames * window, nFFT))
    magSpectr = magSpectr / (len(magSpectr) - 1)

    return magSpectr


def PowerSpectrum(magSpectr):
    # 功率譜: 幅級譜轉為功率譜
    # 輸入: 幅級譜,每幀採樣數
    # 輸出: 功率譜, shape:(分幀數,每幀採樣數/2+1)
    powSpectr = magSpectr ** 2

    return powSpectr


def MelFilterBanks(rate, nFFT, nBanks=26, lowFreq=0, highFreq=None):
    # nBanks: 22 ~ 40, std is 26
    # -------------------------------------------------- #
    # 梅爾濾波器組: 根據最高與最低頻率創建26組濾波器
    # 輸入: 採樣率,每幀採樣數,(濾波器組數,最低頻率,最高頻率)
    # 輸出: 濾波器組, shape:(濾波器組數,每幀採樣數/2+1)
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    if highFreq == None:
        highFreq = rate / 2

    highMel = hz2mel(highFreq)
    lowMel = hz2mel(lowFreq)
    melPoints = np.linspace(lowMel, highMel, nBanks + 2)
    hzPoints = mel2hz(melPoints)
    filterBins = np.intc(hzPoints * (nFFT + 1) / rate)
    filterBanks = np.zeros((nBanks, nFFT // 2 + 1))

    for m in range(0, nBanks):
        for k in range(filterBins[m], filterBins[m + 1]):
            filterBanks[m, k] = (k - filterBins[m]) / (filterBins[m + 1] - filterBins[m])
        for k in range(filterBins[m + 1], filterBins[m + 2]):
            filterBanks[m, k] = (filterBins[m + 2] - k) / (filterBins[m + 2] - filterBins[m + 1])

    return filterBanks


def EnergyFilterBanks(powFrames, filterBanks):
    # 能量濾波器組: 功率譜與濾波器組點積並求對數(10log10)
    # 輸入: 功率譜,濾波器組
    # 輸出: 對數能量濾波器組(dB), shape:(分幀數,濾波器組數)
    energyFB = np.dot(powFrames, np.transpose(filterBanks))

    # if 0 in energyFB:
    #     print('....')

    logEnergyFB = np.log10(energyFB)

    return logEnergyFB


def MFCCs(logEnergyFB, start_nCeps=1, end_nCeps=13):
    # 梅爾頻率倒譜係數: 對對數能量濾波器組做二型離散餘弦變換(DCT-II)
    # 輸入: 對數能量濾波器組,(起始倒譜序號,結束倒譜序號)
    # 輸出: 梅爾頻率倒譜, shape:(分幀數,濾波器組數)
    mfcc = dct(logEnergyFB, type=2, axis=1)[:, start_nCeps:end_nCeps]

    return mfcc


def funcMFCC(audiofile, rtnParm=None,
             preEmphasis=False, preCoeff=0.97,
             frameTime=0.025, overlap=0.6,
             nBanks=26, lowFreq=0, highFreq=None,
             start_nCeps=1, end_nCeps=13, butter=False):
    # MFCC函數: 求梅爾頻率倒譜系列參數
    # 輸入: 音訊檔,(欲返回參數,啟閉預強調,預強調係數,每幀時長,重疊率,
    #              濾波器組數,最低頻率,最高頻率,起始倒譜序號,結束倒譜序號)
    # 輸出: [擇一] 返回參數清單,音訊,預強調音訊,分幀音訊,幅級譜,對數幅級譜,
    #              功率譜,對數功率譜,濾波器組,對數能量濾波器組,梅爾頻率倒譜

    partialExecution = False
    partialExecution = True

    rate, signal = read(audiofile)

    if butter:
        signal = butter_worth(signal)

    if preEmphasis == True:
        PEsignal = PreEmphasis(signal, preCoeff)
    else:
        PEsignal = signal

    """
    資料的多層一維離散小波轉換。

    Parameters
    ----------
    data：類似數組
    輸入資料
    wavelet: 小波物件或名稱字串
    要使用的小波
    mode：str，optional
    訊號擴展模式，請參閱: ref:`Modes < ref - modes > `。
    level：int，optional
    分解等級（必須 >= 0）。 如果等級為「無」（預設），那麼它將使用“dwt_max_level”函數計算。
    axis：int，optional
    用於計算DWT的軸。 如果沒有給出，則使用最後一個軸。
    
    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
    係數數組的有序列表，其中“n”表示分解等級。 第一個元素結果的 (``cA_n``) 是近似係數數組，且以下元素（``cD_n`` - ``cD_1``）是細節係數數組。

    # 進行小波變換
    coeffs = pywt.wavedec(PEsignal, 'db1', level=4)  # 'db1' 是一種小波基，可以根據需要更换

    # 將小波係數重構回信號
    PEsignal = pywt.waverec(coeffs, 'db1')

    for item in coeffs:
        plt.clf()
        ax = plt.subplot()
        ax.plot(item)
        plt.show()
        
    """

    if partialExecution and rtnParm in ['signal', 'PEsignal']:
        return PEsignal

    frames = Framing(PEsignal, rate, frameTime, overlap)

    if partialExecution and rtnParm == 'frames':
        return frames

    magSpectr = FourierTransform(frames)

    if partialExecution and rtnParm == 'magSpectr':
        return magSpectr
    elif partialExecution and rtnParm == 'logMagSpectr':
        return 10 * np.log10(magSpectr)

    powSpectr = PowerSpectrum(magSpectr)

    if partialExecution and rtnParm == 'powSpectr':
        return powSpectr
    elif partialExecution and rtnParm == 'logPowSpectr':
        return 10 * np.log10(powSpectr)

    filterBanks = MelFilterBanks(rate, len(frames[0]), nBanks, lowFreq, highFreq)

    if partialExecution and rtnParm == 'filterBanks':
        return filterBanks

    logEnergyFB = EnergyFilterBanks(powSpectr, filterBanks)

    if partialExecution and rtnParm == 'logEnergyFB':
        return logEnergyFB

    mfcc = MFCCs(logEnergyFB, start_nCeps, end_nCeps)

    if partialExecution and rtnParm == 'mfcc':
        return mfcc

    mfccList = ['list', 'signal', 'PEsignal', 'frames', 'magSpectr', 'logMagSpectr',
                'powSpectr', 'logPowSpectr', 'filterBanks', 'logEnergyFB', 'mfcc']
    parmList = [mfccList, signal, PEsignal, frames, magSpectr, 10 * np.log10(magSpectr),
                powSpectr, 10 * np.log10(powSpectr), filterBanks, logEnergyFB, mfcc]

    for rtn in range(len(mfccList)):
        if rtnParm == mfccList[rtn]:
            return parmList[rtn]

    return mfcc


def butter_worth(data):
    fs = 500
    channel = 1000

    lowCut = 1
    highCut = 30
    nyquist = 0.5 * fs
    low = lowCut / nyquist
    high = highCut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')

    output = signal.filtfilt(b, a, data)

    return output
