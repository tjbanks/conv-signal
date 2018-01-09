import sys, math

from scipy.signal import butter, lfilter, hilbert
import scipy.io as sio
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class Signal:

    sig = None

    def __init__(self, filename='', matvarname=''):
        if filename:
            self.load_file(filename,matvarname)
        return

    def load_file(self, filename, matvarname=''):
        
        mat_extension = ".mat" #To generalize later if we have different filetypes
        x = [] #elements
        if(filename.endswith(mat_extension)):
            if not matvarname:
                raise ValueError('matvarname variable not defined... Exiting')
        matx = load_mat_file(filename, matvarname)
        matfs = matx.size #Number of samples


    def to_supervised(self, ):
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

def load_mat_file(filename, variableName):
    mat = sio.loadmat(filename)
    mdata = mat[variableName]
    return np.array(mdata).ravel()


def hilbert_transform(signal):
    return
