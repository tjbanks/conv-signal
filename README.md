# Processing signals using Keras
This repository will serve as a testing ground for signal processing.

0. Literature review
1. Generate reasonable data with obvious predictors
2. Compile simple convolutional network to verify feasibilty
3. Test features of LIME for explainable AI


## sig2learn.py
A generalized signal to supervised learning class


Any instanciated Signals object that loaded an input file becomes of type
MultiSignal. While MultiSignal hasn't executed [to_np_signal()] will be in
the form (signal, time) and has the following methods available:




Any MultiSignal object that has executed [to_np_signals()] becomes of type
NP_MultiSignal, data will be in the form (signal, time) and has
the following methods available:


After executing [to_supervised_series()] on an NP_MultiSignal object
the object will be in the form (time, look_back*num_signals)


