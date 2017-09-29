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

import importlib as il


#%% .mod passthrough

class modPassThrough():
    """
    Pass calls to .fit and .predict to .mod.fit and .mod.predict
    Handy for any layer of abstraction away from actual keras model
    eg. multiChannelMod.fit() -> convMod.multiChan.fit() -> keras.fit()
    Can call AVMod.fit() instead of AVmod.mod.mod.fit()
    """
    def fit(self, *args, **kwargs):
        return self.mod.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.mod.predict(*args, **kwargs)
    
    def name(self):
        return self.mod.name
    

#%% Functions

class dataHelpers():
    
    def __init__(self, fn, name=''):
        self.fn = fn
        self.name = name
        self.data = dict()

    
    def loadMat(self):
        f = sio.loadmat(self.fn)
        events = f['events'].astype(np.int16)
        sounds = f['sounds'].astype(np.float16)
        rates = MMS(f['rates'].squeeze())
        
        return events, sounds, rates
    
    
    def loadMatAV(self, plotOn=True, idx=[]):
        f = sio.loadmat(self.fn)
    
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
    
        self.eventsA = eventsA
        self.soundsA = soundsA
        self.ratesA = ratesA
        self.decA = decA
        self.eventsV = eventsV
        self.soundsV = soundsV
        self.ratesV = ratesV
        self.decV = decV
        
        if plotOn:
            self.plotAV()
        
        return self
    
    
    def plotAV(self, idx=[], note=''):
        dataHelpers.plot(self.eventsA, self.soundsA, self.ratesA, self.decA, 
                         note='Aud')
        dataHelpers.plot(self.eventsV, self.soundsV, self.ratesV, self.decV, 
                         note='Vis')
    
    
    @staticmethod    
    def plot(events, sounds, rates, dec, idx=[], note=''):
        if idx==[]:
                idx = np.random.randint(events.shape[0])
            
        plt.plot(sounds[idx,:])
        plt.plot(events[idx,:])
        plt.show()
        print(note)
        print(rates[idx])
        print(dec[idx])

    
    def plotDist(self, idx):
        
        print('Mean event distribution in aud set:')
        plt.plot(np.mean(np.abs(self.soundsA[idx,:]), axis=0))
        plt.plot(np.mean(self.eventsA[idx,:], axis=0))
        plt.plot(np.mean(self.yTrainAUd[idx,:], axis=0))
        plt.show()

        print('Mean event distribution in vis set:')
        plt.plot(np.mean(np.abs(self.soundsV[idx,:]), axis=0))
        plt.plot(np.mean(self.eventsV[idx,:], axis=0))
        plt.plot(np.mean(self.yTrainVis[idx,:], axis=0))
        plt.show()

 
    def split(self, n=250):
        for s in ['Aud', 'Vis']:
            if s=='Aud':
                self.idxTrainAud, self.idxTestAud, \
                self.xTrainAud, self.xTrainExpAud, self.yTrainAud, \
                self.yTrainExpAud, self.yTrainRAud, self.yTrainDAud, \
                self.xTestAud, self.xTestExpAud, self.yTestAud, \
                self.yTestExpAud, self.yTestRAud, self.yTestDAud = \
                    dataHelpers.simpleSplit(self.soundsA, 
                                            self.eventsA, 
                                            self.ratesA, 
                                            self.decA, 
                                            n=n)
            elif s=='Vis':
                self.idxTrainVis, self.idxTestVis, \
                self.xTrainVis, self.xTrainExpVis, self.yTrainVis, \
                self.yTrainExpVis, self.yTrainRVis, self.yTrainDVis, \
                self.xTestVis, self.xTestExpVis, self.yTestVis, \
                self.yTestExpVis, self.yTestRVis, self.yTestDVis = \
                    dataHelpers.simpleSplit(self.soundsV, 
                                            self.eventsV, 
                                            self.ratesV, 
                                            self.decV, 
                                            n=n)
                
        return self
    
    
    @staticmethod
    def simpleSplit(sounds, events, rates, dec, n=350):
        """
        Split in to test and train sets, assumes already shuffled.
        Returns multiple shapes of data for convenience
        """
        
        idxTrain = np.zeros(sounds.shape[0])
        idxTrain[0:n] = 1
        idxTrain = idxTrain.astype(np.bool)
        idxTest = np.zeros(sounds.shape[0])
        idxTest[n::] = 1
        idxTest = idxTest.astype(np.bool)
        
        xTrain = sounds[idxTrain,:]
        yTrain = events[idxTrain,:]
        yTrainR = rates[idxTrain]
        yTrainD = dec[idxTrain,:]
        
        xTest = sounds[idxTest,:]
        yTest = events[idxTest,:]
        yTestR = rates[idxTest]
        yTestD = dec[idxTest,:]
        
        # Needed when extracting sequence from LSTM layers
        xTrainExp = np.expand_dims(xTrain, axis=2)
        xTestExp = np.expand_dims(xTest, axis=2)
        yTrainExp = np.expand_dims(yTrain, axis=2)
        yTestExp = np.expand_dims(yTest, axis=2)
    
        return idxTrain, idxTest, \
            xTrain, xTrainExp, yTrain, yTrainExp, yTrainR, yTrainD, xTest, \
            xTestExp, yTest, yTestExp, yTestR, yTestD
            
            
    def trainSet(self, w='Aud'):
        if w=='Aud':
            return self.xTrainAud, self.xTrainExpAud, self.yTrainAud, \
                    self.yTrainExpAud, self.yTrainRAud, self.yTrainDAud
        elif w=='Vis':
            return self.xTrainVis, self.xTrainExpVis, self.yTrainVis, \
                    self.yTrainExpVis, self.yTrainRVis, self.yTrainDVis
    
    
    def testSet(self, w='Aud'):
        if w=='Aud':
            return self.xTestAud, self.xTestExpAud, self.yTestAud, \
                    self.yTestExpAud, self.yTestRAud, self.yTestDAud
        elif w=='Vis':
            return self.xTestVis, self.xTestExpVis, self.yTestVis, \
                    self.yTestExpVis, self.yTestRVis, self.yTestDVis
        
            
#%% Single channel model class

class singleChannelMod(modPassThrough):
    def __init__(self, mod, dataLength=512, **kwargs):
        self.mod = mod(dataLength, **kwargs)
        self.results = dict()
        
   
    def evaluate(self, dataSet, setName='train', layerName='', trans=True):
        
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
        
        mod = self.mod
        # xTrainExpAud = np.expand_dims(self.xTrainAud, 2)
        
        xTrainAud, xTrainExpAud, yTrainAud, yTrainExpAud, yTrainRAud, yTrainDAud \
            = dataSet
        
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
            intOut = interMod.predict(np.expand_dims(xTrainExpAud[0,:,:], 
                                                     axis=0))
    
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
        
        # Claculate accuracies
        decAcc = np.sum(yTrainDAud[:,1] == (audDec[:,1]>0.5))\
                        /yTrainDAud.shape[0]
        rateAcc = np.sum(yTrainRAud == np.round(audRate))\
                        /yTrainDAud.shape[0]
        # Calculate rate loss                        
        rateLoss = np.mean(abs(audRate.squeeze() - yTrainRAud.squeeze()))
        
        print('Overall rate loss:', rateLoss)
        print('Overall dec acc:', decAcc)
        
        self.results[setName+'RateLoss'] = rateLoss 
        self.results[setName+'RateAcc'] = rateAcc
        self.results[setName+'DecAcc'] = decAcc
        
        return self
    
    def printComp(self, setName=['train', 'test'], note=''):    
        for s in setName:
            self.printResult(setName=s, note=note)
            
              
    def printResult(self, setName='train', note=''):
        print('--'*20)
        print(self.name())
        print(note)
        print('   '+setName+':')
        print('      Rate Loss:', str(self.results[setName+'RateLoss']), 
              'Rate Acc: '+str(np.round(self.results[setName+'RateAcc'],2))+'%')
        print('      Dec Acc: '+str(np.round(self.results[setName+'DecAcc'],2))+'%')



#%% Multi-channel model class
 
class multiChannelMod(modPassThrough):
    def __init__(self, mod, dataLength=512, **kwargs):
        
        self.mod = mod(dataLength, **kwargs)
        self.results = dict()
        
    
    def evaluate(self, dataSet, setName='train', layerName='', trans=True):
        
        """
        Eval and plot for model with
        inputs=[audInput, visInput],
        outputs=[audRateOutput, visRateOutput, audRateOutput, AVDecOutput]
        eg. ConvModels.multiChan
        
        - Plots a random single example
        - And named layer output
        - Then overall loss and accuracy for set
        
        TODO:
            - Add support of variable number of model outputs
        """
        
        # Preallocate the following possible metrics
        # Will depend on actual model outputs
        rateLossA = np.nan
        rateLossV = np.nan
        rateLossAV = np.nan
        rateAccA = np.nan
        rateAccV = np.nan
        rateAccAV = np.nan
        decAccA = np.nan
        decAccV = np.nan
        decAccAV = np.nan
        
        mod = self.mod
        
        xTrainAud, xTrainExpAud, yTrainAud, \
        yTrainExpAud, yTrainRAud, yTrainDAud \
            = dataSet.trainSet(w='Aud')
            
        xTrainVis, xTrainExpVis, yTrainVis, \
        yTrainExpVis, yTrainRVis, yTrainDVis \
            = dataSet.trainSet(w='Vis')
        
        # Predict from model
        audRate, visRate, AVRate, AVDec = \
            self.predict([xTrainExpAud, xTrainExpVis])
        
        print('Random example:')
        idx = np.random.randint(audRate.shape[0])
        print('Stim, events:')    
        plt.plot(xTrainAud[idx,:])
        plt.plot(xTrainVis[idx,:])
        plt.plot(yTrainAud[idx,:])
        plt.plot(yTrainVis[idx,:])
        plt.show()
        
        # Also get output from conv -> flatten for this example
        if layerName != '':
            print(layerName)      
            interMod = Model(inputs=mod.input, 
                             outputs=mod.get_layer(layerName).output)
            intOut = interMod.predict(np.expand_dims(xTrainExpAud[0,:,:], 
                                                     axis=0))
    
            print(layerName, 'output:')
            if trans:
                plt.plot(np.transpose(intOut.squeeze()))
            else:
                plt.plot(intOut.squeeze())
                
            plt.show()
            print('Layer output shape:')
            print(intOut.shape)
    
        print('Pred rate A | V:', audRate[idx], '|', visRate[idx])
        print('GT rate A | V:', yTrainRAud[idx], '|', yTrainRAud[idx])
        print('Pred dec AV:', AVDec[idx])
        
        
        # Claculate accuracies
        # Add: decAccA
        # Add: decAccV
        decAccAV = np.sum(yTrainDAud[:,1] == (AVDec[:,1]>0.5))\
                        /yTrainDAud.shape[0]
                        
        rateAccA = np.sum(yTrainRAud == np.round(audRate))\
                        /yTrainDAud.shape[0]
        rateAccV = np.sum(yTrainRVis == np.round(visRate))\
                        /yTrainDAud.shape[0]
        rateAccAV = np.sum(yTrainRAud == np.round(AVRate))\
                        /yTrainDAud.shape[0]                
        # Calculate rate loss                        
        rateLossA = np.mean(abs(audRate.squeeze() - yTrainRAud.squeeze()))
        rateLossV = np.mean(abs(visRate.squeeze() - yTrainRVis.squeeze()))
        rateLossAV = np.mean(abs(AVRate.squeeze() - yTrainRAud.squeeze()))
        
        # Print results
        print('Overall rate loss A | V | AV:', 
              np.round(rateLossA, 2), '|', 
              np.round(rateLossV, 2), '|', 
              np.round(rateLossAV, 2))
        print('Overall acc loss A | V | AV:', 
              np.round(rateAccA, 2), '|', 
              np.round(rateAccV, 2), '|', 
              np.round(rateAccAV, 2))
        print('Overall dec acc A | V | AV:', 
              np.round(decAccA, 2), '|', 
              np.round(decAccV, 2), '|', 
              np.round(decAccAV, 2))
        
        # Save results
        self.results[setName+'rateLossA'] = rateLossA
        self.results[setName+'rateLossV'] = rateLossV
        self.results[setName+'rateLossAV'] = rateLossAV
        self.results[setName+'rateAccA'] = rateAccA
        self.results[setName+'rateAccV'] = rateAccV
        self.results[setName+'rateAccAV'] = rateAccAV
        self.results[setName+'decAccA '] = decAccA 
        self.results[setName+'decAccV']= decAccV
        self.results[setName+'decAccAV'] = decAccAV
        
        return self

    
    def printComp(self, setName=['train', 'test'], note=''):    
        for s in setName:
            self.printResult(setName=s, note=note)
            
              
    def printResult(self, setName='train', note=''):
        print('--'*20)
        print(self.name())
        print(note)
        print('   '+setName+':')
        print('      Rate Loss:', str(self.results[setName+'RateLoss']), 
              'Rate Acc: '+str(np.round(self.results[setName+'RateAcc'],2))+'%')
        print('      Dec Acc: '+str(np.round(self.results[setName+'DecAcc'],2))+'%')
        