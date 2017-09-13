# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:44:42 2017

@author: Gareth
"""

#%% Imports

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Embedding, concatenate
from keras.layers import Conv1D
from keras import optimizers

import importlib as il
import utils
il.reload(utils)
from utils import singleChannelHelpers


#%%

class ConvModels(singleChannelHelpers):
    def __init__(self, name=''):
        self.results = dict()
        self.name = name
        pass

    def simple(self, x1, nDims=128, ks=512, strides=256):
        """
        1 Channel (A or V)
        Rate and decision output
        inputs=[audInput],
        outputs=[audRateOutput, audDecOutput]
        
        TODO:
            Rename aud references to something more generic - can be A or V
        """
        
        # Prepare inputs
        x1Width = x1.shape[1] # Also aud lstm output width
        
        # Create Input layers
        inp = Input(shape=(x1Width,1), dtype='float32', name='input')
        
        # Aud LSTM    
        conv = Conv1D(nDims, kernel_size= ks, strides=strides, 
                         input_shape=(x1Width,1), 
                         name='audConv_l1')(inp)
        conv = Flatten(name='audConv_l2')(conv) 
        conv = Dropout(0.3, name='audConv_l3')(conv) 
        
        # Aud dense layers
        a = Dense(int(x1Width/2), activation='relu', 
                  name='rate_l1')(conv)
        a = Dropout(0.15, name='rate_l2')(a)
        rateOutput = Dense(1, activation='relu', 
                              name='rateOutput')(a)
        
        # And make decision
        decOutput = Dense(2, activation='softmax', 
                              name='decOutput')(rateOutput)
        
        # Make model with 1 input and 2 outputs
        model = Model(inputs=[inp],
                      outputs=[rateOutput, decOutput])
        
        # Complile with weighted losses
        model.compile(optimizer='rmsprop', loss='mse',
                      loss_weights=[0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self
    