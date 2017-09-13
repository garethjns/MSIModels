# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:59:22 2017

@author: Gareth


Helper functions for loading data and evalusating models
TODO:
    - Split in to three classes to inherit in to model classes
        - Helpers (use AV loader, single not needed)
        - Single channel helpers
        - Dual channel helpers
"""

#%% Imports

import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import minmax_scale as MMS, OneHotEncoder

from keras.models import Model


#%% Functions

def loadMat(fn):
    f = sio.loadmat(fn)
    events = f['events'].astype(np.int16)
    sounds = f['sounds'].astype(np.float16)
    rates = MMS(f['rates'].squeeze())
    
    return events, sounds, rates


def loadMatAV(fn, plotOn=True, idx=[]):
    f = sio.loadmat(fn)

    # Aud
    eventsA = f['eventsA'].astype(np.int16)
    soundsA = f['soundsA'].astype(np.float16)
    ratesA = f['ratesA']
    # One-hot decision
    oh = OneHotEncoder()
    decA = (ratesA>np.mean(ratesA)).astype(np.int16)
    oh = oh.fit(decA)
    decA = oh.transform(decA).toarray()
    
    # Vis
    eventsV = f['eventsV'].astype(np.int16)
    soundsV = f['soundsV'].astype(np.float16)
    ratesV = f['ratesV']
    # One-hit decision using existing scheme
    decV = (ratesV>np.mean(ratesV)).astype(np.int16)
    decV = oh.transform(decV).toarray()

    if plotOn:
        if idx==[]:
            idx = np.random.randint(eventsA.shape[0])
        plt.plot(soundsA[idx,:])
        plt.plot(eventsA[idx,:])
        plt.show()
        print(ratesA[idx])
        print(decA[idx])
        
        plt.plot(soundsV[idx,:])
        plt.plot(eventsV[idx,:])
        plt.show()
        print(ratesV[idx])
        print(decV[idx])
        
    return eventsA, soundsA, ratesA, decA, eventsV, soundsV, ratesV, decV


def simpleSplit(sounds, events, rates, dec, n=350):
    """
    Split in to test and train sets, assumes already shuffled.
    Returns multiple shapes of data for convenience
    """
    xTrain = sounds[0:n,:]
    yTrain = events[0:n,:]
    yTrainR = rates[0:n]
    yTrainD = dec[0:n,:]
    
    xTest = sounds[n::,:]
    yTest = events[n::,:]
    yTestR = rates[n::]
    yTestD = dec[n::,:]
    
    # Needed when extracting sequence from LSTM layers
    xTrainExp = np.expand_dims(xTrain, axis=2)
    xTestExp = np.expand_dims(xTest, axis=2)
    yTrainExp = np.expand_dims(yTrain, axis=2)
    yTestExp = np.expand_dims(yTest, axis=2)

    return xTrain, xTrainExp, yTrain, yTrainExp, yTrainR, yTrainD, \
            xTest, xTestExp, yTest, yTestExp, yTestR, yTestD
            

def evalSingleChanMod(mod, 
              xTrainAud, yTrainAud, yTrainRAud, yTrainDAud, 
              soundsAud, eventsAud, layerName='', trans=True):
    
    """
    Eval and plot for model with
    inputs=[audInput],
    outputs=[audRateOutput, audDecOutput]
    eg. ConvModels.simpleConv1D and LSTMModels.simpleLSTM
    
    - Plots a random single example
    - And named layer output
    - Then overall loss and accuracy for set
    
    TODO:
        - Rename references to Aud - can be A or V
        - Rename references to train - can be any set with labels
    """
    
    xTrainExpAud = np.expand_dims(xTrainAud, 2)
    
    # Predict from model
    audRate, audDec = \
        mod.predict(xTrainExpAud)
    

    print('Random single example:')
    idx = np.random.randint(audRate.shape[0])
    print('Stim, events:')    
    plt.plot(xTrainAud[idx,:])
    plt.plot(yTrainAud[idx,:])
    plt.show()
    
    # Also get output from conv -> flatten for this example
    if layerName != '':
        print(layerName)      
        interMod = Model(inputs=mod.input, 
                         outputs=mod.get_layer(layerName).output)
        intOut = interMod.predict(np.expand_dims(xTrainExpAud[0,:,:], axis=0))

        print(layerName, 'output:')
        if trans:
            plt.plot(np.transpose(intOut.squeeze()))
        else:
            plt.plot(intOut.squeeze())
            
        plt.show()
        print('Layer output shape:')
        print(intOut.shape)

    print('Pred rate:', audRate[idx])
    print('Pred dec:', audDec[idx])
    print('GT:', yTrainRAud[idx])
    
    print('Mean event distribution in set:')
    plt.plot(np.mean(np.abs(soundsAud), axis=0))
    plt.plot(np.mean(eventsAud, axis=0))
    plt.plot(np.mean(yTrainAud, axis=0))
    plt.show()
    
    acc = np.sum(yTrainDAud[:,1] == (audDec[:,1]>0.5))/yTrainDAud.shape[0]
    loss = np.mean(abs(audRate.squeeze() - yTrainRAud.squeeze()))
    print('Overall rate loss:', loss)
    print('Overall dec acc:', acc)
    
    return loss, acc

              
def printLossAcc(trainLoss, trainAcc, testLoss, testAcc, note=''):
    print('--'*20)
    print(note)
    print('   Train:')
    print('      Loss:', str(trainLoss)+', Acc;', np.round(trainAcc,2))
    print('Test:')
    print('      Loss:', str(testLoss)+', Acc;', np.round(testAcc,2))
    print('--'*20)


def evalAVMod(mod, xTrainAud, yTrainAud, yTrainRAud, yTrainDAud, soundsAud, eventsAud, 
              xTrainVis, yTrainVis, yTrainRVis, yTrainDVis, soundsVis, eventsVis):
    """
    Not fully implemented yet
    
    
    Eval and plot for model AV input and 5 outputs
    inputs=[audInput, visInput],
    outputs=[audLSTMOutput, audRateOutput, 
             visLSTMOutput, visRateOutput,
             AVRate, AVDec]
    
    eg. LSTMModels.lateAccumComplex and LSTMModels.earlyAccumComplex

    TODO:
        - Reimplement in similar way to evalSingleChanMod

    """
    print('Fix me first')          
    return
    
    xTrainExpAud = np.expand_dims(xTrainAud, 2)
    xTrainExpVis = np.expand_dims(xTrainVis, 2)
    
    audLSTM, audRate, visLSTM, visRate, AVRate, AVDec = \
        mod.predict([xTrainExpAud, xTrainExpVis])
    

    print('TRAIN:')
    idx = np.random.randint(audRate.shape[0])
    
    plt.plot(xTrainAud[idx,:])
    plt.plot(yTrainAud[idx,:])
    plt.plot(audLSTM[idx,:])
    plt.show()
    plt.plot(xTrainVis[idx,:])
    plt.plot(yTrainVis[idx,:])
    plt.plot(visLSTM[idx,:])
    plt.show()
    
    print('Aud:.')
    plt.plot(np.mean(np.abs(soundsAud), axis=0))
    plt.plot(np.mean(eventsAud, axis=0))
    plt.plot(np.mean(yTrainAud, axis=0))
    plt.show()

    print('Pred:', audRate[idx])
    print('GT:', yTrainRAud[idx])
    
    print('Vis:.')
    plt.plot(np.mean(np.abs(soundsVis), axis=0))
    plt.plot(np.mean(yTrainVis, axis=0))
    plt.plot(np.mean(eventsVis, axis=0))
    plt.show()

    print('Pred:', visRate[idx])
    print('GT:', yTrainRVis[idx])
    print('AV:')
    print('Pred:', AVRate[idx])
    print('Dec:', AVDec[idx])
    print('GT:', yTrainRVis[idx], '|', yTrainDVis[idx])
    print('Overall train loss A:', 
          np.mean(abs(audRate.squeeze() - yTrainRAud.squeeze())))
    print('Overall train loss V:', 
          np.mean(abs(visRate.squeeze() - yTrainRVis.squeeze())))
    print('Overall train loss AV:', 
          np.mean(abs(AVRate.squeeze() - yTrainRAud.squeeze())))
    print('Overall train dec acc AV:', 
          np.sum(yTrainDVis[:,1] == (AVDec[:,1]>0.5))/yTrainDVis.shape[0])
    
    audLSTM, audRate, visLSTM, visRate, AVRate, AVDec = \
        mod.predict([xTestExpAud, xTestExpVis])
    
    print('TEST:')
    idx = np.random.randint(audRate.shape[0])
    
    plt.plot(xTestAud[idx,:])
    plt.plot(yTestAud[idx,:])
    plt.plot(audLSTM[idx,:])
    plt.show()
    plt.plot(xTestVis[idx,:])
    plt.plot(yTestVis[idx,:])
    plt.plot(visLSTM[idx,:])
    plt.show()
    
    print('Aud:.')
    plt.plot(np.mean(np.abs(soundsAud), axis=0))
    plt.plot(np.mean(eventsAud, axis=0))
    plt.plot(np.mean(yTrainAud, axis=0))
    plt.show()
    print('Pred:', audRate[idx])
    print('GT:', yTestRAud[idx])
    
    print('Vis:')
    plt.plot(np.mean(np.abs(soundsVis), axis=0))
    plt.plot(np.mean(yTrainVis, axis=0))
    plt.plot(np.mean(eventsVis, axis=0))
    print('Pred:', visRate[idx])
    print('GT:', yTestRVis[idx])
    print('AV:')
    print('Pred:', AVRate[idx])
    print('Dec:', AVDec[idx])
    print('GT:', yTestRVis[idx], '|', yTestDVis[idx])
    
    print('Overall test loss A:', 
          np.mean(abs(audRate.squeeze() - yTestRAud.squeeze())))
    print('Overall test loss V:', 
          np.mean(abs(visRate.squeeze() - yTestRVis.squeeze())))
    print('Overall test loss AV:', 
          np.mean(abs(AVRate.squeeze() - yTestRAud.squeeze())))
    print('Overall test dec acc AV:', 
          np.sum(yTestDVis[:,1] == (AVDec[:,1]>0.5))/yTestDVis.shape[0])