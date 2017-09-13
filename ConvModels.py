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


def simpleConv1D(x1Aud, nDims=128, ks=512, strides=256):
    """
    1 Channel (A or V)
    Rate and decision output
    inputs=[audInput],
    outputs=[audRateOutput, audDecOutput]
    
    TODO:
        Rename aud references to something more generic - can be A or V
    """
    
    # Prepare inputs
    x1AudWidth = x1Aud.shape[1] # Also aud lstm output width
    
    # Create Input layers
    audInput = Input(shape=(x1AudWidth,1), dtype='float32', name='audInput')
    
    # Aud LSTM    
    audConv = Conv1D(nDims, kernel_size= ks, strides=strides, 
                     input_shape=(x1AudWidth,1), 
                     name='audConv_l1')(audInput)
    audConv = Flatten(name='audConv_l2')(audConv) 
    audConv = Dropout(0.3, name='audConv_l3')(audConv) 
    
    # Aud dense layers
    a = Dense(int(x1AudWidth/2), activation='relu', 
              name='audRate_l1')(audConv)
    a = Dropout(0.15, name='audRate_l2')(audConv)
    audRateOutput = Dense(1, activation='relu', 
                          name='audRateOutput')(a)
    
    # And make decision
    audDecOutput = Dense(2, activation='softmax', 
                          name='audDecOutput')(audRateOutput)
    
    # Make model with 1 input and 2 outputs
    model = Model(inputs=[audInput],
                  outputs=[audRateOutput, audDecOutput])
    
    # Complile with weighted losses
    model.compile(optimizer='rmsprop', loss='mse',
                  loss_weights=[0.5, 0.5], 
                  metrics=['accuracy'])

    print(model.summary())
    return model