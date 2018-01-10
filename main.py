from sig2learn import Signals
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler


matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"

s = Signals().load_mat(matfile,matvar).set_name('LFP1').cut(1000,4000) \
    .append_bandpass(2,10).to_hilbert() \
    .append_bandpass(11,20).to_hilbert() \
    .append_bandpass(21,30).to_hilbert() \
    .append_bandpass(31,40).to_hilbert() \
    .append_bandpass(41,50).to_hilbert() \
    .append_bandpass(51,63).to_hilbert() \
    .append_bandpass(64,84).to_hilbert() \
    .remove_original_signal()
    
scaler = MinMaxScaler(feature_range=(0,1))
    
nps = s.to_np_signals().scale(scaler).to_supervised_series(300, 20)
print s.to_np_signals().get_np().shape
print nps.shape
train = nps[:3000, :]
test = nps[3000:, :]
# split into input and outputs
trainX, trainY = train[:, :-1], train[:, -1]
testX, testY =    test[:, :-1],  test[:, -1]
#print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    
# reshape input to be [channels, time steps, features]
temp = []   
def myfunc(x):
    temp.append(np.reshape(x, (7, 300, 1)))
np.apply_along_axis(myfunc, axis=1, arr=trainX )
trainX = np.array(temp);

print trainX.shape
sys.exit('ya')

print len(s.get_signals())

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
    plot_signal(sig.get_np_arr(), 1000, sig.get_name(), newPlot=False)
#plot_signal(s.get_signals()[1].get_np_arr(), 1000, s.get_signals()[1].get_name())