# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:44:42 2017

@author: Gareth
"""

#%% Imports

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers import Conv1D
from keras import optimizers

import importlib as il
from old import utils

il.reload(utils)


#%%

class ConvModels(utils.modPassThrough):
    def __init__(self, name=''):
        self.results = dict()
        self.name = name
        pass

    def simple(self, dataLength=512, nFil=128, ks=512, strides=256):
        """
        1 Channel (A or V)
        Rate and decision output
        inputs=[audInput],
        outputs=[audRateOutput, audDecOutput]
        
        TODO:
            Rename aud references to something more generic - can be A or V
        """
        
        # Prepare inputs
        xLen = dataLength # Also aud lstm output width
        
        # Create Input layers
        inp = Input(shape=(xLen, 1), dtype='float32', name='input')
        
        # Aud LSTM    
        conv = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen, 1), 
                         name='audConv_l1')(inp)
        conv = Flatten(name='audConv_l2')(conv) 
        conv = Dropout(0.3, name='audConv_l3')(conv) 
        
        # Aud dense layers
        a = Dense(int(xLen/2), activation='relu', 
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
        rmsprop = optimizers.rmsprop(lr=0.0001)
        model.compile(optimizer=rmsprop, loss='mse',
                      loss_weights=[0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self
    
    def multiChanLate(self, xLen, nFil=128, ks=512, strides=256):
        """
        2 Channel (A and V)
        Rate and decision output
        inputs=[audInput, visInput],
        outputs=[ARate, VRate, AVRate, AVDec]
        """
        
        # Prepare inputs
        # xLen = x1.shape[1] # Channel 1 length
        # For model creation only need one channel length. Can pad with keras 
        # for unisensory input, or use noise for "off" channel.
        
        # Create Input layers - channel 1
        inp1 = Input(shape=(xLen,1), dtype='float32', name='c1_input')
        
        # Chan 1 conv    
        conv1 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c1_Conv_l1')(inp1)
        conv1 = Flatten(name='c1_Conv_l2')(conv1) 
        conv1 = Dropout(0.3, name='c1_Conv_l3')(conv1) 
        
        # Chan 1 dense layers
        a1 = Dense(int(xLen/2), activation='relu', 
                  name='c1_rate_l1')(conv1)
        a1 = Dropout(0.15, name='c1_rate_l2')(a1)
        rateOutput1 = Dense(1, activation='relu', 
                              name='c1_rateOutput')(a1)
        
        # Chan 2 input
        inp2 = Input(shape=(xLen,1), dtype='float32', name='c2_input')
        
        # Chan 2 conv    
        conv2 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c2_conv_l1')(inp2)
        conv2 = Flatten(name='c2_conv_l2')(conv2) 
        conv2 = Dropout(0.3, name='c2_conv_l3')(conv2) 
        
        # Chan 2 dense layers
        a2 = Dense(int(xLen/2), activation='relu', 
                  name='c2_rate_l1')(conv2)
        a2 = Dropout(0.15, name='c2_rate_l2')(a2)
        rateOutput2 = Dense(1, activation='relu', 
                              name='c2_rateOutput')(a2)
        
        # Integrate
        AV = concatenate([a1, a2], name='AVRate_l1')
        rateOutputAV = Dense(1, activation='relu', 
                              name='rateOutputAV')(AV)
        
        # And make decision
        decOutput = Dense(2, activation='softmax', 
                              name='decOutput')(rateOutputAV)
        
        # Make model with 1 input and 2 outputs
        model = Model(inputs=[inp1, inp2],
                      outputs=[rateOutput1, rateOutput2, rateOutputAV, decOutput])
        
        # Complile with weighted losses
        rmsprop = optimizers.rmsprop(lr=0.0001)
        model.compile(optimizer=rmsprop, loss='mse',
                      loss_weights=[0.2, 0.2, 0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self
    
    
    def multiChanEarly(self, xLen, nFil=128, ks=512, strides=256):
        """
        2 Channel (A and V)
        Rate and decision output
        inputs=[audInput, visInput],
        outputs=[ARate, VRate, AVRate, AVDec]
        """
        
        # Prepare inputs
        # xLen = x1.shape[1] # Channel 1 length
        # For model creation only need one channel length. Can pad with keras 
        # for unisensory input, or use noise for "off" channel.
        
        # Create Input layers - channel 1
        inp1 = Input(shape=(xLen,1), dtype='float32', name='c1_input')
        
        # Chan 1 conv    
        conv1 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c1_Conv_l1')(inp1)
        conv1 = Flatten(name='c1_Conv_l2')(conv1) 
        conv1 = Dropout(0.3, name='c1_Conv_l3')(conv1) 
        
        # Chan 1 dense layers
        a1 = Dense(int(xLen/2), activation='relu', 
                  name='c1_rate_l1')(conv1)
        b1 = Dropout(0.15, name='c1_rate_l2')(a1)
        rateOutput1 = Dense(1, activation='relu', 
                              name='c1_rateOutput')(b1)
        
        # Chan 2 input
        inp2 = Input(shape=(xLen,1), dtype='float32', name='c2_input')
        
        # Chan 2 conv    
        conv2 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c2_conv_l1')(inp2)
        conv2 = Flatten(name='c2_conv_l2')(conv2) 
        conv2 = Dropout(0.3, name='c2_conv_l3')(conv2) 
        
        # Chan 2 dense layers
        a2 = Dense(int(xLen/2), activation='relu', 
                  name='c2_rate_l1')(conv2)
        b2 = Dropout(0.15, name='c2_rate_l2')(a2)
        rateOutput2 = Dense(1, activation='relu', 
                              name='c2_rateOutput')(b2)
        
        # Integrate
        AV = concatenate([conv1, conv2], name='AVRate_l1')
        AV = Dense(int(xLen), activation='relu', 
                  name='AVRate_l2')(AV)
        AV = Dense(int(xLen/2), activation='relu', 
                  name='AVRate_l3')(AV)
        AV = Dropout(0.15, name='AVRate_l4')(AV)
        rateOutputAV = Dense(1, activation='relu', 
                              name='rateOutputAV')(AV)
        
        # And make decision
        decOutput = Dense(2, activation='softmax', 
                              name='decOutput')(rateOutputAV)
        
        # Make model with 1 input and 2 outputs
        model = Model(inputs=[inp1, inp2],
                      outputs=[rateOutput1, rateOutput2, rateOutputAV, decOutput])
        
        # Complile with weighted losses
        rmsprop = optimizers.rmsprop(lr=0.0001)
        model.compile(optimizer=rmsprop, loss='mse',
                      loss_weights=[0.05, 0.05, 0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self
    
    
    def multiChanEarlyLate(self, xLen, nFil=128, ks=512, strides=256):
        """ WIP
        2 Channel (A and V)
        Rate and decision output
        inputs=[audInput, visInput],
        outputs=[ARate, VRate, AVRate, AVDec]
        """
        
        # Prepare inputs
        # xLen = x1.shape[1] # Channel 1 length
        # For model creation only need one channel length. Can pad with keras 
        # for unisensory input, or use noise for "off" channel.
        
        # Create Input layers - channel 1
        inp1 = Input(shape=(xLen,1), dtype='float32', name='c1_input')
        
        # Chan 1 conv    
        conv1 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c1_Conv_l1')(inp1)
        conv1 = Flatten(name='c1_Conv_l2')(conv1) 
        conv1 = Dropout(0.3, name='c1_Conv_l3')(conv1) 
        
        # Chan 1 dense layers
        a1 = Dense(int(xLen/2), activation='relu', 
                  name='c1_rate_l1')(conv1)
        b1 = Dropout(0.15, name='c1_rate_l2')(a1)
        rateOutput1 = Dense(1, activation='relu', 
                              name='c1_rateOutput')(b1)
        
        # Chan 2 input
        inp2 = Input(shape=(xLen,1), dtype='float32', name='c2_input')
        
        # Chan 2 conv    
        conv2 = Conv1D(nFil, kernel_size= ks, strides=strides, 
                         input_shape=(xLen,1), 
                         name='c2_conv_l1')(inp2)
        conv2 = Flatten(name='c2_conv_l2')(conv2) 
        conv2 = Dropout(0.3, name='c2_conv_l3')(conv2) 
        
        # Chan 2 dense layers
        a2 = Dense(int(xLen/2), activation='relu', 
                  name='c2_rate_l1')(conv2)
        b2 = Dropout(0.15, name='c2_rate_l2')(a2)
        rateOutput2 = Dense(1, activation='relu', 
                              name='c2_rateOutput')(b2)
        
        # Integrate
        AV = concatenate([conv1, conv2, b1, b2], name='AVRate_l1')
        AV = Dense(int(xLen), activation='relu', 
                  name='AVRate_l2')(AV)
        AV = Dense(int(xLen/2), activation='relu', 
                  name='AVRate_l3')(AV)
        AV = Dropout(0.15, name='AVRate_l4')(AV)
        rateOutputAV = Dense(1, activation='relu', 
                              name='rateOutputAV')(AV)
        
        # And make decision
        decOutput = Dense(2, activation='softmax', 
                              name='decOutput')(rateOutputAV)
        
        # Make model with 1 input and 2 outputs
        model = Model(inputs=[inp1, inp2],
                      outputs=[rateOutput1, rateOutput2, rateOutputAV, decOutput])
        
        # Complile with weighted losses
        rmsprop = optimizers.rmsprop(lr=0.0001)
        model.compile(optimizer=rmsprop, loss='mse',
                      loss_weights=[0.05, 0.05, 0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self