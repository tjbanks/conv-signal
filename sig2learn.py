import sys, math

from scipy.signal import butter, lfilter, hilbert
import scipy.io as sio
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

##Input
class Signal:

    sig = None

    def __init__(self):
        return

    def load_mat(self, filename, matvarname):
        
        mat_extension = ".mat" #To generalize later if we have different filetypes
        x = [] #elements
        if(filename.endswith(mat_extension)):
            if not matvarname:
                raise ValueError('matvarname variable not defined... Exiting')
        mat = sio.loadmat(filename)
        mdata = mat[matvarname]
        matx = np.array(mdata).ravel()

        return MultiSignal(SingleSignal(matx))

##Intermediary
class SingleSignal:

    np_arr = None

    def __init__(self, numpy_arr):
        self.np_arr = numpy_arr
        return

    def testprint(self):
        print self.np_arr


##Intermediary
class MultiSignal:

    signals = []

    def __init__(self, singlesignal):
        self.signals.append(singlesignal)
        return


#================ AUX FUNCTIONS ================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def hilbert_transform(signal):
    return
