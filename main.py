import sig2learn
from sig2learn import Signals

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"
look_back = 300
future_element = 20

s = Signals().load_mat(matfile,matvar).set_name('LFP1').cut(1000,21000)
    
scaler = MinMaxScaler(feature_range=(0,1))
    
nps = s.to_np_signals().scale(scaler).to_supervised_series(look_back, future_element, stride=1)


train = nps[:2000, :]
test = nps[2000:, :]
# split into input and outputs
trainX, trainY = train[:, :-1], train[:, -1]
testX, testY =    test[:, :-1],  test[:, -1]

channels = len(s.get_signals())
features = look_back
trainX = sig2learn.supervised_series_to4D(trainX, channels, features)
testX = sig2learn.supervised_series_to4D(testX, channels, features)




def plot_signal(x, n_per_second, signame, newPlot=True):
    
    if newPlot:
        plt.figure(figsize=(15,10))
        plt.clf()
        
    fs = x.size
    t = np.linspace(0, float(fs)/n_per_second, fs, endpoint=False)
    plt.plot(t, x, label=signame)
    plt.legend()
    plt.xlabel('Time (seconds)')

plt.figure(figsize=(15,10))
for sig in s.get_signals():
    plot_signal(sig.get_np_arr(), 1000, sig.get_name(), newPlot=False)#len of signal / fs
#plot_signal(s.get_signals()[1].get_np_arr(), 1000, s.get_signals()[1].get_name())