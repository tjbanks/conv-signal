#!/usr/bin/env python
__author__ = "Tyler Banks"
__version__ = "1.0.0"

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

    def replace_np_arr(self, npa):
        self.np_arr = npa
        return

    def get_cutout(self,n_start,n_end):
        return self.np_arr[n_start:n_end]

    def get_bandpass_filter(self, low, high, ord=2):
        return butter_bandpass_filter(self.np_arr, low, high, self.np_arr.size, order=ord)

    def get_hilbert_transform(self):
        analytic_signal = hilbert(self.np_arr)
        return np.abs(analytic_signal)


##Intermediary
class MultiSignal:

    signals = []

    def __init__(self, singlesignal):
        self.append_signal(singlesignal)
        return

    def append_signal(self, singlesignal):
        self.signals.append(singlesignal)
        return self

    def remove_signal(self, index):
        del self.signals[index]
        return self

    def remove_original_signal(self):
        remove_signal(self, 0)
        return self

    def append_bandpass_filter(self, low, high, order=2, index=0):
        s = SingleSignal(self.signals[index].get_bandpass_filter(low, high, order))
        self.append_signal(s)
        return self

    def replace_with_bandpass_filter(self):
        npa = self.signals[-1].get_bandpass_filter(low, high, order)
        self.signals[-1].replace_np_arr(npa)
        return self

    def replace_with_cutout(self,n_start,n_end):
        npa = self.signals[-1].get_cutout(n_start,n_end)
        self.signals[-1].replace_np_arr(npa)
        return self

    def replace_with_hilbert_transform(self):
        npa = self.signals[-1].get_hilbert_transform()
        self.signals[-1].replace_np_arr(npa)
        return self

    def to_numpy():
        return

    def scale():
        return

    def to_supervised_series_2D():
        """Returns the selected index time series, last element will be goal"""
        return

    def to_supervised_series_3D():
        """Returns all indexes as time series"""
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


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

    
    # convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg
 
    
def series_to_supervised_DNNData(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed

def series_to_supervised_CNNData(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed

