# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:59:40 2018
https://freesound.org/people/Corsica_S/sounds/415067/
@author: Tyler
"""
from sig2learn import Signals

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file = 'test.wav'

plt.figure(figsize=(15,10))
s = Signals().load_wav16(file).set_name('sound1') \
    .cut_seconds(0, 5).plot().write_to_wav16_file('test-cut-orig.wav', scale=False).plot_fft() \
    .append_bandpass(1000,50000).plot(newPlot=True).write_to_wav16_file('test-cut-band1000-50000.wav')
    

