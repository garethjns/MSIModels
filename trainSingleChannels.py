# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:01:41 2017

@author: Gareth
"""

#%% Imports

# Reload libraries that might have been edited
import importlib as il

import LSTMModels
il.reload(LSTMModels)
from LSTMModels import LSTMModels

import ConvModels
il.reload(ConvModels)
from ConvModels import ConvModels

import utils
il.reload(utils)
from utils import dataHelpers as dh

from keras import backend as K

import matplotlib.pyplot as plt


#%% Import data set

dPath = 'Data/'
dSet = 'stimData_AV_s12_20000x400.mat'

# Load the dataset
data = dh(dPath+dSet, name='s12')
data = data.loadMatAV()

# Split the data set in to test and train
# Returns:
# self.
# xTrainAud, xTrainExpAud, yTrainAud, yTrainExpAud, yTrainRAud, yTrainDAud, \
# xTestAud, xTestExpAud, yTestAud, yTestExpAud, yTestRAud, yTestDAud
# Which are...
# x with and without exanded dimensions
# y (for A, V sequences) with and without expanded dimensions
# y (for AV rate - assumes equal rate for A and V at the moment) as scale
# y (for AV decision) as binary
data = data.split(n=12000)

data.plotDist(idx = data.idxTrainVis)
data.plotDist(idx = data.idxTestVis)


#%% Fit two independent single channel models
"""
Convolutional

Example run with stimData_AV_s11_10000x400.mat and settings:
    nDims=256, ks=128, strides=32
    batch_size=150, epochs=75, validation_split=0.2
----------------------------------------
A channel, conv
   Train:
      Loss: 1.44146, Acc; 0.84
Test:
      Loss: 1.49312, Acc; 0.84
----------------------------------------
----------------------------------------
V channel, conv
   Train:
      Loss: 0.596947, Acc; 0.85
Test:
      Loss: 0.595911, Acc; 0.84
----------------------------------------

"""

K.clear_session()

modAConv = ConvModels(name='AConv').simple(data.xTrainExpAud, 
                                   nDims=256, ks=128, strides=32)
modVConv = ConvModels(name='VConv').simple(data.xTrainExpVis, 
                                   nDims=256, ks=128, strides=32)

# Note assuming all matached rates here so AVRate and AVDec same between 
# modalities (just using aud here:)
historyA = modAConv.mod.fit(data.xTrainExpAud,
            [data.yTrainRAud, data.yTrainDAud], # AV
            batch_size=500, epochs=5000, validation_split=0.2)
historyV = modVConv.mod.fit(data.xTrainExpVis,
            [data.yTrainRVis, data.yTrainDVis], # AV
            batch_size=500, epochs=5000, validation_split=0.2)

plt.plot(historyA.history['loss'])
plt.plot(historyA.history['val_loss'])
plt.show()
plt.plot(historyV.history['loss'])
plt.plot(historyV.history['val_loss'])
plt.show()

# Predict
audRate, audDec = modAConv.mod.predict(data.xTrainExpAud)
# Predict
visRate, visDec = modVConv.mod.predict(data.xTrainExpVis)

modAConv = modAConv.evaluate(data.trainSet('Aud'), setName='train')
modAConv = modAConv.evaluate(data.testSet('Aud'), setName='test')
modVConv = modVConv.evaluate(data.trainSet('Vis'), setName='train')
modVConv = modVConv.evaluate(data.testSet('Vis'), setName='test')

modAConv.printComp()
modVConv.printComp()


#%% Fit two independent single channel models
""" 
LSTM

Example run with stimData_AV_s11_10000x400.mat and settings:
    nDims=256
    batch_size=150, epochs=75, validation_split=0.2
    Both reach 100% accuracy with >100 epochs
----------------------------------------
A channel, LSTM
   Train:
      Loss: 0.253548, Acc; 1.0
Test:
      Loss: 0.270073, Acc; 1.0
----------------------------------------
----------------------------------------
V channel, LSTM
   Train:
      Loss: 0.474363, Acc; 0.84
Test:
      Loss: 0.475426, Acc; 0.84
----------------------------------------

"""

K.clear_session()

modALSTM = LSTMModels(name='ALSTM').simple(data.xTrainExpAud, 
                                   nDims=256)
modVLSTM = LSTMModels(name='VLSTM').simple(data.xTrainExpVis, 
                                   nDims=256)

# Note assuming all matached rates here so AVRate and AVDec same between 
# modalities (just using aud here:)
historyA = modALSTM.mod.fit(data.xTrainExpAud,
            [data.yTrainRAud, data.yTrainDAud], # AV
            batch_size=150, epochs=1, validation_split=0.2)
historyV = modVLSTM.mod.fit(data.xTrainExpVis,
            [data.yTrainRVis, data.yTrainDVis], # AV
            batch_size=150, epochs=1, validation_split=0.2)

plt.plot(historyA.history['loss'])
plt.plot(historyA.history['val_loss'])
plt.show()
plt.plot(historyV.history['loss'])
plt.plot(historyV.history['val_loss'])
plt.show()

# Predict
audRate, audDec = modALSTM.mod.predict(data.xTrainExpAud)
# Predict
visRate, visDec = modVLSTM.mod.predict(data.xTrainExpVis)

# Full training set - A
modALSTM = modALSTM.evaluate(data.trainSet('Aud'), setName='train')
modALSTM = modALSTM.evaluate(data.testSet('Aud'), setName='test')
modVLSTM = modVLSTM.evaluate(data.trainSet('Vis'), setName='train')
modVLSTM = modVLSTM.evaluate(data.testSet('Vis'), setName='test')

modALSTM.printComp()
modVLSTM.printComp()


#%% Print comparison

modAConv.printComp()
modVConv.printComp()
modALSTM.printComp()
modVLSTM.printComp()


#%% Fit a multichannel model
# Convolutional


#%% Fit a multichannel model
# LSTM
K.clear_session()

mod = LSTMModels.lateAccum(xTrainExpAud, xTrainExpVis, nPts=12)

# Note assuming all matached rates here so AVRate and AVDec same between 
# modalities (just using aud here:)
history = mod.fit([xTrainAud, xTrainExpVis],
            [yTrainAud, yTrainRAud, # Aud
             yTrainVis, yTrainRVis, # Vis
             yTrainRAud, yTrainDAud], # AV
            batch_size=1, epochs=50, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# Predict
audLSTM, audRate, visLSTM, visRate, AVRate, AVDec = \
mod.predict([xTrainExpAud, xTrainExpVis])
