# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:54:31 2017

@author: Gareth
"""

#%% Imports

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Embedding, concatenate
from keras.layers import LSTM
from keras import optimizers

import importlib as il
import utils
il.reload(utils)


#%% Models

class LSTMModels(utils.modPassThrough):
    def __init__(self, name=''):
        self.results = dict()
        self.name = name

    def seqLSTM(self, dataLength=512, nPts=128):
        
        timeSteps = dataLength
        
        model = Sequential()
        model.add(Embedding(timeSteps, output_dim=timeSteps))
        model.add(LSTM(nPts))
        model.add(Dropout(0.5))
        model.add(Dense(timeSteps, activation='relu'))
        
        model.compile(loss='mse',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
        return model
    
    
    def simple(self, dataLength=512, nDims=128):
        """
        1 Channel (A or V)
        Rate and decision output
        inputs=[audInput],
        outputs=[audRateOutput, audDecOutput]
        
        TODO:
            Rename aud references to something more generic - can be A or V
        """
        
        # Prepare inputs
        x1Len = dataLength # Also aud lstm output width
        
        # Create Input layers
        inp = Input(shape=(x1Len, 1), dtype='float32', name='input')
        
        # Aud LSTM    
        lstm = LSTM(nDims, input_shape=(x1Len, 1), 
                       return_sequences=True, name='LSTM_l1')(inp)
        lstm = Flatten(name='LSTM_l2')(lstm) 
        lstm = Dropout(0.3, name='LSTM_l3')(lstm) 
        
        # Aud dense layers
        a = Dense(int(x1Len/2), activation='relu', 
                  name='rate_l1')(lstm)
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
    
    
    def lateAccum(self, dataLength=512, nPts=128):
        """
        Seperate sensory processing
        Seperate accumulation and decision
        Combined for AV rate and dec 
        
        inputs=[audInput, visInput],
        outputs=[audLSTMOutput, audRateOutput, 
                 visLSTMOutput, visRateOutput,
                 AVRate, AVDec]
        """
        
        # Prepare inputs
        xLen = dataLength
        
        # Create Input layers
        audInput = Input(shape=(xLen, 1), dtype='float32', name='audInput')
        visInput = Input(shape=(xLen, 1), dtype='float32', name='visInput')
        
        # Aud LSTM    
        audLSTM = LSTM(nPts, input_shape=(xLen,1), 
                       return_sequences=True, name='audLSTM_l1')(audInput)
        audLSTM = Flatten(name='audLSTM_l2')(audLSTM) 
        audLSTM = Dropout(0.3, name='audLSTM_l3')(audLSTM) 
        audLSTMOutput = Dense(xLen, name='audLSTMOutput', 
                              activation='relu')(audLSTM)
        
        # Vis LSTM    
        visLSTM = LSTM(nPts, input_shape=(xLen,1), 
                       return_sequences=True, name='visLSTM_l1')(visInput)
        visLSTM = Dropout(0.3, name='visLSTM_l2')(visLSTM) 
        visLSTM = Flatten(name='visLSTM_l3')(visLSTM) 
        visLSTMOutput = Dense(xLen, name='visLSTMOutput', 
                              activation='relu')(visLSTM)
        
        # Aud dense layers
        a = Dense(int(xLen/2), activation='sigmoid', 
                  name='audRate_l1')(audLSTMOutput)
        a = Dropout(0.15, name='audRate_l2')(a)
        audRateOutput = Dense(1, activation='relu', 
                              name='audRateOutput')(a)
        
        # Vis dense layers
        v = Dense(int(xLen/2), activation='sigmoid', 
                  name='visRate_l1')(visLSTMOutput)
        v = Dropout(0.15, name='visRate_l2')(v)
        visRateOutput = Dense(1, activation='relu', 
                              name='visRateOutput')(v)
        
        # Late concatenation of estimates
        AV = concatenate([audRateOutput, visRateOutput], name='AVRateConcat')
        AVRate = Dense(1, activation='relu', 
                              name='AVRateOutput')(AV)
        
        # And make decision
        AVDec = Dense(2, activation='softmax', 
                              name='AVDecOutput')(AVRate)
        
        # Make model with 1 input and 2 outputs
        model = Model(inputs=[audInput, visInput],
                      outputs=[audLSTMOutput, audRateOutput, 
                               visLSTMOutput, visRateOutput,
                               AVRate, AVDec])
        
        # Complile with weighted losses
        model.compile(optimizer='rmsprop', loss='mse',
                      loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                      metrics=['accuracy'])
    
        print(model.summary())
        
        self.mod = model
        
        return self


#%% Others - add later
# LSTM, late accumulation
def lateAccumComplex(x1Aud, x1Vis, nPts=128):
    # Seperate sensory processing
    # Seperate accumulation and decision
    # Combined for AV rate and dec 

    # Prepare inputs
    x1AudWidth = x1Aud.shape[1] # Also aud lstm output width
    x1VisWidth = x1Vis.shape[1] # Also vis lstm output width
      
    audInput = Input(shape=(x1AudWidth,1), dtype='float32', name='audInput')
    visInput = Input(shape=(x1VisWidth,1), dtype='float32', name='visInput')
    
    # Aud LSTM    
    audLSTM = LSTM(nPts, input_shape=(x1AudWidth,1), 
                   return_sequences=True, name='audLSTM_l1')(audInput)
    audLSTM = Dropout(0.1, name='audLSTM_l2')(audLSTM) 
    audLSTM = LSTM(int(nPts/2), return_sequences=True, 
                   input_shape=(x1AudWidth,1), name='audLSTM_l3')(audLSTM)
    audLSTM = Flatten(name='audLSTM_l4')(audLSTM) 
    audLSTM = Dropout(0.4, name='audLSTM_l5')(audLSTM) 
    audLSTMOutput = Dense(x1AudWidth, name='audLSTMOutput', 
                          activation='sigmoid')(audLSTM)
    
    # Vis LSTM    
    visLSTM = LSTM(nPts, input_shape=(x1VisWidth,1), 
                   return_sequences=True, name='visLSTM_l1')(visInput)
    visLSTM = Dropout(0.1, name='visLSTM_l2')(visLSTM)
    visLSTM = LSTM(int(nPts/2), return_sequences=True, 
                   input_shape=(x1VisWidth,1), name='visLSTM_l3')(visLSTM)
    visLSTM = Dropout(0.4, name='visLSTM_l4')(visLSTM) 
    visLSTM = Flatten(name='visLSTM_l5')(visLSTM) 
    visLSTMOutput = Dense(x1AudWidth, name='visLSTMOutput', 
                          activation='sigmoid')(visLSTM)
    
    # Aud dense layers
    a = Dense(int(x1AudWidth/2), activation='sigmoid', 
              name='audRate_l1')(audLSTMOutput)
    a = Dropout(0.15, name='audRate_l2')(a)
    a = Dense(int(x1AudWidth/4), activation='sigmoid', 
              name='audRate_l3')(a)
    audRateOutput = Dense(1, activation='linear', 
                          name='audRateOutput')(a)
    
    # Vis dense layers
    v = Dense(int(x1VisWidth/2), activation='sigmoid', 
              name='visRate_l1')(visLSTMOutput)
    v = Dropout(0.15, name='visRate_l2')(v)
    v = Dense(int(x1VisWidth/4), activation='sigmoid', 
              name='visRate_l3')(v)
    visRateOutput = Dense(1, activation='linear', 
                          name='visRateOutput')(v)
    
    # Late concatenation of estimates
    AV = concatenate([audRateOutput, visRateOutput], name='AVRateConcat')
    AVRate = Dense(1, activation='relu', 
                          name='AVRateOutput')(AV)
    
    # And make decision
    AVDec = Dense(2, activation='softmax', 
                          name='AVDecOutput')(AVRate)
    
    # Make model with 1 input and 2 outputs
    model = Model(inputs=[audInput, visInput],
                  outputs=[audLSTMOutput, audRateOutput, 
                           visLSTMOutput, visRateOutput,
                           AVRate, AVDec])
    
    # Complile with weighted losses
    model.compile(optimizer='rmsprop', loss='mse',
                  loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                  metrics=['accuracy'])

    print(model.summary())
    return model


# LSTM, early accumulation
def earlyAccumComplex(x1Aud, x1Vis, nPts=128):
    # Seperate sensory processing
    # Seperate accumulation and decision
    # Combined for AV rate and dec 

    # Prepare inputs
    x1AudWidth = x1Aud.shape[1] # Also aud lstm output width
    x1VisWidth = x1Vis.shape[1] # Also vis lstm output width
      
    audInput = Input(shape=(x1AudWidth,1), dtype='float32', name='audInput')
    visInput = Input(shape=(x1VisWidth,1), dtype='float32', name='visInput')
    
    # Aud LSTM    
    audLSTM = LSTM(nPts, input_shape=(x1AudWidth,1), 
                   return_sequences=True, name='audLSTM_l1')(audInput)
    audLSTM = LSTM(int(nPts/2), return_sequences=True, 
                   input_shape=(x1AudWidth,1), name='audLSTM_l2')(audLSTM)
    audLSTM = Flatten(name='audLSTM_l3')(audLSTM) 
    audLSTMOutput = Dense(x1AudWidth, name='audLSTMOutput', 
                          activation='sigmoid')(audLSTM)
    
    # Vis LSTM    
    visLSTM = LSTM(nPts, input_shape=(x1VisWidth,1), 
                   return_sequences=True, name='visLSTM_l1')(visInput)
    visLSTM = LSTM(int(nPts/2), return_sequences=True, 
                   input_shape=(x1VisWidth,1), name='visLSTM_l2')(visLSTM)
    visLSTM = Flatten(name='visLSTM_l3')(visLSTM) 
    visLSTMOutput = Dense(x1AudWidth, name='visLSTMOutput', 
                          activation='sigmoid')(visLSTM)
    
    # Keep A and V dense layers to get A and V rate estimates, 
    # but weight these losses as very low
    # Aud dense layers
    audRate_l1 = Dense(int(x1AudWidth/2), activation='sigmoid', 
              name='audRate_l1')(audLSTMOutput)
    audRate_l2 = Dense(int(x1AudWidth/4), activation='sigmoid', 
              name='audRate_l2')(audRate_l1)
    audRateOutput = Dense(1, activation='linear', 
                          name='audRateOutput')(audRate_l2)
    
    # Vis dense layers
    visRate_l1 = Dense(int(x1VisWidth/2), activation='sigmoid', 
              name='visRate_l1')(visLSTMOutput)
    visRate_l2 = Dense(int(x1VisWidth/4), activation='sigmoid', 
              name='visRate_l2')(visRate_l1)
    visRateOutput = Dense(1, activation='linear', 
                          name='visRateOutput')(visRate_l2)
    
    # Early concatenation of estimates
    AV = concatenate([audLSTMOutput, visLSTMOutput], name='AVLSTMConcat')
    # Then do similar prcoessing as for single channels
    # (1 extra layer)
    AVRate_l1 = Dense(int(x1AudWidth), activation='sigmoid', 
              name='AVRate_l1')(AV)
    AVRate_l2 = Dense(int(x1AudWidth/2), activation='sigmoid', 
              name='AVRate_l2')(AVRate_l1)
    AVRate_l3 = Dense(int(x1AudWidth/4), activation='sigmoid', 
              name='AVRate_l3')(AVRate_l2)
    AVRate = Dense(1, activation='relu', 
                          name='AVRate')(AVRate_l3)
    
    # And make decision
    AVDec = Dense(2, activation='softmax', 
                          name='AVDec')(AVRate)
    
    # Make model with 1 input and 2 outputs
    model = Model(inputs=[audInput, visInput],
                  outputs=[audLSTMOutput, audRateOutput, 
                           visLSTMOutput, visRateOutput,
                           AVRate, AVDec])
    
    # Complile with weighted losses
    model.compile(optimizer='rmsprop', loss='mse',
                  loss_weights=[0.4, 0., 0.4, 0., 0.75, 0.75], 
                  metrics=['accuracy'])

    print(model.summary())
    return model
