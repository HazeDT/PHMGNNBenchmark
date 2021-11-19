import numpy as np


def FFT(x):
    '''
    :param x: the raw signal
    :return: the signal after FFT
    '''
    x = np.fft.fft(x)
    x = np.abs(x) / len(x)
    x = x[range(int(x.shape[0] / 2))]
    return x

def add_noise(x, snr):
    '''
    :param x: the raw siganl
    :param snr: the signal to noise ratio
    :return: noise signal
    '''
    d = np.random.randn(len(x))  # generate random noise
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal