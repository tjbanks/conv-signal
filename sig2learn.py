#!/usr/bin/env python
__author__ = "Tyler Banks"
__version__ = "1.0.0"

from scipy.signal import butter, lfilter, hilbert
import scipy.io as sio
import numpy as np
import pandas as pd
from pandas import DataFrame

##Input
class Signals(object):

    def __init__(self):
        return

    def load_mat(self, filename, matvarname):
        matx = load_mat(filename, matvarname)
        return MultiSignal(SingleSignal(matx))

##Intermediary
class SingleSignal(object):

    #np_arr = None

    def __init__(self, numpy_arr):
        self.np_arr = numpy_arr
        self.readable_name = ''
        return 

    def testprint(self):
        print self.np_arr

    def get_np_arr(self):
        return self.np_arr
    
    def get_name(self):
        return self.readable_name
    
    def replace_np_arr(self, npa):
        self.np_arr = npa
        return
    
    def set_name(self, name):
        self.readable_name = name

    def get_cutout(self,n_start,n_end):
        return self.np_arr[n_start:n_end]

    def get_bandpass_filter(self, low, high, ord=2):
        return butter_bandpass_filter(self.np_arr, low, high, self.np_arr.size, order=ord)

    def get_hilbert_transform(self):
        analytic_signal = hilbert(self.np_arr)
        return np.abs(analytic_signal)


##Intermediary
class MultiSignal(object):

    #signals = []

    def __init__(self, singlesignal):
        self.signals = [] 
        self.append_signal(singlesignal)
        return
    
    def get_signals(self):
        return self.signals

    def append_signal(self, singlesignal):
        self.signals.append(singlesignal)
        return self

    def remove_signal(self, index):
        del self.signals[index]
        return self

    def remove_original_signal(self):
        self.remove_signal(0)
        return self
    
    def set_name(self, name):
        self.signals[-1].set_name(name)
        return self
    
    def cut(self,n_start,n_end):
        npa = self.signals[-1].get_cutout(n_start,n_end)
        self.signals[-1].replace_np_arr(npa)
        return self

    def load_mat(self, filename, matvarname):
        matx = load_mat(filename, matvarname)
        self.append_signal(SingleSignal(matx))
        return self
        
    def append_bandpass(self, low, high, order=2, index=0):
        s = SingleSignal(self.signals[index].get_bandpass_filter(low, high, order))
        s.set_name('%s Filter Band %d - %d Hz'%(self.signals[index].get_name(), low, high))
        self.append_signal(s)
        return self

    def to_bandpass(self, low, high, order=2):
        s = self.signals[-1]
        npa = s.get_bandpass_filter(low, high, order)
        s.set_name('Filter Band %d - %d Hz'%(low, high))
        self.signals[-1].replace_np_arr(npa)
        return self
    
    def append_hilbert(self, index=-1):
        s = SingleSignal(self.signals[index].get_hilbert_transform())
        s.set_name('%s Hilbert Transform'%(self.signals[index].get_name()))
        self.append_signal(s)
        return self

    def to_hilbert(self):
        s = self.signals[-1]
        npa = self.signals[-1].get_hilbert_transform()
        s.set_name('%s Hilbert Transform'%(s.get_name()))
        self.signals[-1].replace_np_arr(npa)
        return self

    def to_np_signals(self):
        bandsets = []
        for s in self.signals:
            bandsets.append(s.get_np_arr())
        return NP_MultiSignal(np.array(bandsets))

class NP_MultiSignal(object):
    
    #np_arr = []
    
    def __init__(self, npa):
        self.np_arr = npa
        return
    
    def get_np(self):
        return self.np_arr
    
    def scale(self, scaler):
        self.np_arr = self.np_arr.astype('float32')
        self.np_arr = self.np_arr.transpose()
        self.np_arr = scaler.fit_transform(self.np_arr)
        return self

    def to_supervised_series(self, look_back, future_element, stride=1):
        """Returns all indexes as time series, last element of each line is the desired output
        The desired output is given as +future_element of the last row"""
        out = series_to_supervised_Data(self.np_arr, n_in=look_back, n_out=future_element, dropnan=True)
        out = out.values[::stride,:]
        return out

    
#================ AUX FUNCTIONS ================

def load_mat(filename, matvarname):
    mat_extension = ".mat" #To generalize later if we have different filetypes
    if(filename.endswith(mat_extension)):
        if not matvarname:
            raise ValueError('matvarname variable not defined... Exiting')
    mat = sio.loadmat(filename)
    mdata = mat[matvarname]
    matx = np.array(mdata).ravel()
    return matx

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

def series_to_supervised_Data(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed


def supervised_series_to4D(data, channels, features):
    # reshape input to be [channels, time steps, features]
    temp = []   
    def myfunc(x):
        temp.append(np.reshape(x, (channels, features, 1)))
    np.apply_along_axis(myfunc, axis=1, arr=data )
    ret = np.array(temp);
    return ret
