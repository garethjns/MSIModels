# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:47:42 2017

@author: Gareth
"""

#%% Imports
# Reload libraries that might have been edited
import importlib as il

from old import LSTMModels, utils, ConvModels

il.reload(LSTMModels)
from old.LSTMModels import LSTMModels

il.reload(ConvModels)
from old.ConvModels import ConvModels

il.reload(utils)
from old.utils import dataHelpers as dh
from old.utils import multiChannelMod as AVMod

from keras import backend as K

#%% Load AV dataset

dPath = 'Data/'
dSet = 'stimData_AV_s15_20000x400.mat'

# Load the dataset
data = dh(dPath+dSet, name='s15')
data = data.loadMatAV()

data = data.split(nTrain=12000)

data.plotDists()


#%% Model set up

# Save path
mPath = 'Models/'
# Number of epochs
nEpConv = 1500
nEpLSTM = 200


#%% Train late integration (Conv, AV data)

# K.clear_session()

name = 'A_VConv_Late'
modAVConv1 = AVMod(mod=ConvModels(name=name).multiChanLate,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nFil=256, ks=128, strides=32)

modAVConv1.history = modAVConv1.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=10000, epochs=nEpConv, validation_split=0.2, verbose=0)

modAVConv1.plotHistory()

# Evaluate
modAVConv1 = modAVConv1.evaluate(data, setName='train')
modAVConv1 = modAVConv1.evaluate(data, setName='test')

# Save to disk
modAVConv1.save()


#%% Test load
# History not saved

# modAVConv1_tmp = AVMod(mod=ConvModels(name=name).multiChanLate)
# modAVConv1_tmp = modAVConv1_tmp.load()


#%% Train early integration (Conv, AV data)

K.clear_session()

name = 'A_VConv_Early'
modAVConv2 = AVMod(mod=ConvModels(name=name).multiChanEarly,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nFil=256, ks=128, strides=32)

modAVConv2.history = modAVConv2.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=10000, epochs=nEpConv, validation_split=0.2, verbose=0)

modAVConv2.plotHistory()

# Evaluate
modAVConv2 = modAVConv2.evaluate(data, setName='train')
modAVConv2 = modAVConv2.evaluate(data, setName='test')

# Save to disk
modAVConv2.save()


#%% Train late integration (LSTM, AV data)

K.clear_session()

name = 'A_VLSTM_Late'
modAVLSTM = AVMod(mod=LSTMModels(name=name).multiChanLate,  
                  dataLength = data.xTrainExpAud.shape[1], 
                  nPts=128)

modAVLSTM.history = modAVLSTM.fit([data.xTrainExpAud, data.xTrainExpVis],
            [data.yTrainRAud, data.yTrainRAud, 
             data.yTrainRAud, data.yTrainDAud],
            batch_size=1000, epochs=nEpLSTM, validation_split=0.2, verbose=0)

modAVLSTM.plotHistory()

# Evaluate
modAVLSTM = modAVLSTM.evaluate(data, setName='train')
modAVLSTM = modAVLSTM.evaluate(data, setName='test')

# Save to disk
modAVLSTM.save()
