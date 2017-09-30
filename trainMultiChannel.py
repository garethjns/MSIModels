# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:47:42 2017

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
from utils import multiChannelMod as AVMod

from keras import backend as K

import matplotlib.pyplot as plt


#%% Load AV dataset

dPath = 'Data/'
dSet = 'stimData_AV_s14_20000x400.mat'

# Load the dataset
data = dh(dPath+dSet, name='s12')
data = data.loadMatAV()

data = data.split(n=12000)

data.plotDist(idx = data.idxTrainVis)
data.plotDist(idx = data.idxTestVis)


#%% Train late integration (Conv, AV data)

K.clear_session()

modAVConv1 = AVMod(mod=ConvModels(name='AVConv').multiChanLate,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nFil=256, ks=128, strides=32)

historyAV = modAVConv1.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=500, epochs=500, validation_split=0.2)

plt.plot(historyAV.history['loss'])
plt.plot(historyAV.history['val_loss'])
plt.show()


# Evaluate
modAVConv1 = modAVConv1.evaluate(data, setName='train')
modAVConv1 = modAVConv1.evaluate(data, setName='test')


#%% Train early integration (Conv, AV data)

K.clear_session()

modAVConv2 = AVMod(mod=ConvModels(name='AVConv').multiChanEarly,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nFil=256, ks=128, strides=32)

historyAV = modAVConv2.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=500, epochs=1000, validation_split=0.2)

plt.plot(historyAV.history['loss'])
plt.plot(historyAV.history['val_loss'])
plt.show()


# Evaluate
modAVConv2 = modAVConv2.evaluate(data, setName='train')
modAVConv2 = modAVConv2.evaluate(data, setName='test')


#%% Train late integration (LSTM, AV data)

K.clear_session()

modAVLTM = AVMod(mod=LSTMModels(name='AVLSTM').multiChan,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nPts=128)

historyAV = modAVLTM.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=200, epochs=5, validation_split=0.2)

plt.plot(historyAV.history['loss'])
plt.plot(historyAV.history['val_loss'])
plt.show()


# Evaluate
modAVConv = modAVConv.evaluate(data, setName='train')
modAVConv = modAVConv.evaluate(data, setName='test')
