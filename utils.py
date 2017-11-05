# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:59:22 2017

@author: Gareth


Helper functions for loading data and evalusating models


TODO:
    - Update single channel mods with plotHistory, save/load functions,
    figure saving
"""

#%% Imports

import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import minmax_scale as MMS, OneHotEncoder

from keras.models import Model

import importlib as il


import os
from keras.models import load_model


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
    

class modelsGeneral():
    """
    Methods shared between multi and single channel models
    """
    def save(self):
        if os.path.isdir(self.path)==False:
            os.mkdir(self.path)
        
        fn = self.path+'mod.mod'
        print('Saving:', fn)
        self.mod.mod.save(fn)
    
    def load(self):
        
        fn = self.path+'mod.mod'
        print('Loading:', fn)
        self.mod.mod = load_model(fn)
        
        return self
    
    
    def prepDir(self):
        
        self.path = 'Models/'+ self.name() + '/'
        
        if os.path.isdir(self.path)==False:
            os.mkdir(self.path)
        
        self.hgPath = self.path+'History/'
        if os.path.isdir(self.hgPath)==False:
            os.mkdir(self.hgPath)
        
        self.egPath = self.path+'Eval/'
        if os.path.isdir(self.egPath)==False:
            os.mkdir(self.egPath)
            
        return self
    

#%% Single channel model class

class singleChannelMod(modPassThrough, modelsGeneral):
    
    def __init__(self, mod, dataLength=512, **kwargs):
        
        self.mod = mod(dataLength, **kwargs)
        self.results = dict()
        self.path = 'Models/'+ self.name() + '/'
        
        if os.path.isdir(self.path)==False:
            os.mkdir(self.path)
   
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
        plt.plot(xTrainAud[idx,:], label = '"c1" raw')
        plt.plot(yTrainAud[idx,:], label = '"c1" events')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Mag.')
        plt.savefig(self.egPath+'Random single example_' + setName + '.png')
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
 
class multiChannelMod(modPassThrough, modelsGeneral):
    def __init__(self, mod, dataLength=512, **kwargs):
        
        self.mod = mod(dataLength, **kwargs)
        self.results = dict()
        self.history=[]
        
        self = self.prepDir()
                
        
    def plotHistory(self, which=['', 'val']):
        
        if self.history != []:
            for w in which:
                if w == '':
                    print('Training history')
                    fApp = 'train'
                else:
                    print('Validation history')
                    fApp = w
                    w = w+'_'
                    
                plt.semilogy(self.history.history[w+'loss'],
                         label=w+'loss')
                plt.semilogy(self.history.history[w+'c1_rateOutput_loss'], 
                         label=w+'c1_rateOutput_loss')
                plt.semilogy(self.history.history[w+'c2_rateOutput_loss'], 
                         label=w+'c2_rateOutput_loss')
                plt.semilogy(self.history.history[w+'rateOutputAV_loss'], 
                         label=w+'rateOutputAV_loss')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Rate losses')
                plt.savefig(self.hgPath+'Rate losses_' + fApp + '.png')
                plt.show()
                
                plt.plot(self.history.history[w+'decOutput_loss'],
                         label=w+'decOutput_loss')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Decision loss')
                plt.savefig(self.hgPath+'Dec losses_' + fApp + '.png')
                plt.show()
                
                plt.plot(self.history.history[w+'c1_rateOutput_acc'],
                         label=w+'c1_rateOutput_acc')
                plt.plot(self.history.history[w+'c2_rateOutput_acc'],
                         label=w+'c2_rateOutput_acc')
                plt.plot(self.history.history[w+'rateOutputAV_acc'],
                         label=w+'rateOutputAV_acc')
                plt.plot(self.history.history[w+'decOutput_acc'],
                         label=w+'decOutput_acc')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.savefig(self.hgPath+'Accuracy_' + fApp + '.png')
                plt.show()                
    
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
        plt.plot(xTrainAud[idx,:], label = 'Aud raw')
        plt.plot(xTrainVis[idx,:], label = 'Vis raw')
        plt.plot(yTrainAud[idx,:], label = 'Aud events')
        plt.plot(yTrainVis[idx,:], label = 'Vis events')
        plt.legend
        plt.xlabel('Time')
        plt.ylabel('Mag.')
        plt.savefig(self.egPath+'Random single example_' + setName + '.png')
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
        print('Overall rate acc A | V | AV:', 
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
        
#%% Data class

class dataHelpers():
    
    def __init__(self, fn, name=''):
        self.fn = fn
        self.name = name
        self.data = dict()
        self.trainIdx = []
        self.testIdx = []
    
    
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
        
        self.type = f['type'].astype(np.float16)
        self.eventsA = eventsA
        self.soundsA = soundsA
        self.ratesA = ratesA
        self.decA = decA
        self.eventsV = eventsV
        self.soundsV = soundsV
        self.ratesV = ratesV
        self.decV = decV
        self.n = self.ratesA.shape[0]
        
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

    
    def plotDists(self):
        
        print('Mean event distribution in aud train set:')
        plt.plot(np.mean(np.abs(self.xTrainAud), axis=0))
        plt.plot(np.mean(self.yTrainAud, axis=0))
        plt.show()
        
        print('Mean event distribution in aud test set:')
        plt.plot(np.mean(np.abs(self.xTestAud), axis=0))
        plt.plot(np.mean(self.yTestAud, axis=0))
        plt.show()
        
        print('Mean event distribution in vis train set:')
        plt.plot(np.mean(np.abs(self.xTrainVis), axis=0))
        plt.plot(np.mean(self.yTrainVis, axis=0))
        plt.show()
        
        print('Mean event distribution in vis test set:')
        plt.plot(np.mean(np.abs(self.xTestVis), axis=0))
        plt.plot(np.mean(self.yTestVis, axis=0))
        plt.show()
        
        
    def split(self, nTrain=250):
        randIdx = np.random.choice(range(0, self.n), self.n, replace=False)
        self.trainIdx = randIdx[0:nTrain]
        self.testIdx = randIdx[nTrain::]
    
        return self
    
    
    def __getattribute__(self, name):
        """
        Return data using self.trainIdx and self.testIdx splits
        """
        # Type
        if (name=='typeTrain') | (name=='trainType'):
            return self.type[self.trainIdx]
        if (name=='typeTest') | (name=='testType'):
            return self.type[self.testIdx]
        # Aud
        elif (name=='xTrainAud') | (name=='xTrainA') | (name=='xTrainC1'):
            return self.soundsA[self.idxTrain,:]
        elif (name=='xTrainExpAud') | (name=='xTrainExpA') | (name=='xTrainExpC1'):
            return np.expand_dims(self.soundsA[self.idxTrain,:], axis=2)
        elif (name=='yTrainAud') | (name=='yTrainA') | (name=='yTrainC1'):
            return self.eventsA[self.idxTrain,:]
        elif (name=='yTrainExpAud') | (name=='yTrainExpA') | (name=='yTrainExpC1'):
            return np.expand_dims(self.eventsA[self.idxTrain,:], axis=2)
        elif (name=='yTrainRAud') | (name=='yTrainRA') | (name=='yTrainRC1'):
            return self.ratesA[self.idxTrain]
        elif (name=='yTrainDAud') | (name=='yTrainDA') | (name=='yTrainDC1'):
            return self.decA[self.idxTrain]
        elif (name=='xTestAud') | (name=='xTestA') | (name=='xTestC1'):
            return self.soundsA[self.idxTest,:]
        elif (name=='xTestExpAud') | (name=='xTestExpA') | (name=='xTestExpC1'):
            return np.expand_dims(self.soundsA[self.idxTest,:], axis=2)
        elif (name=='yTestAud') | (name=='yTestA') | (name=='yTestC1'):
            return self.eventsA[self.idxTest,:]
        elif (name=='yTestExpAud') | (name=='yTestExpA') | (name=='yTestExpC1'):
            return np.expand_dims(self.eventsA[self.idxTest,:], axis=2)
        elif (name=='yTestRAud') | (name=='yTestRA') | (name=='yTestRC1'):
            return self.ratesA[self.idxTest]
        elif (name=='yTestDAud') | (name=='yTestDA') | (name=='yTestDC1'):
            return self.decA[self.idxTest]
        # Vis
        if (name=='xTrainVis') | (name=='xTrainV') | (name=='xTrainC2'):
            return self.soundsV[self.idxTrain,:]
        elif (name=='xTrainExpVis') | (name=='xTrainExpV') | (name=='xTrainExpC2'):
            return np.expand_dims(self.soundsV[self.idxTrain,:], axis=2)
        elif (name=='yTrainVis') | (name=='yTrainV') | (name=='yTrainC2'):
            return self.eventsV[self.idxTrain,:]
        elif (name=='yTrainExpVis') | (name=='yTrainExpV') | (name=='yTrainExpC2'):
            return np.expand_dims(self.eventsV[self.idxTrain,:], axis=2)
        elif (name=='yTrainRVis') | (name=='yTrainRV') | (name=='yTrainRC2'):
            return self.ratesV[self.idxTrain]
        elif (name=='yTrainDVis') | (name=='yTrainDV') | (name=='yTrainDC2'):
            return self.decV[self.idxTrain]
        elif (name=='xTestVis') | (name=='xTestV') | (name=='xTestC2'):
            return self.soundsV[self.idxTest,:]
        elif (name=='xTestExpVis') | (name=='xTestExpV') | (name=='xTestExpC2'):
            return np.expand_dims(self.soundsV[self.idxTest,:], axis=2)
        elif (name=='yTestVis') | (name=='yTestV') | (name=='yTestC2'):
            return self.eventsV[self.idxTest,:]
        elif (name=='yTestExpVis') | (name=='yTestExpV') | (name=='yTestExpC2'):
            return np.expand_dims(self.eventsV[self.idxTest,:], axis=2)
        elif (name=='yTestRVis') | (name=='yTestRV') | (name=='yTestRC2'):
            return self.ratesV[self.idxTest]
        elif (name=='yTestDVis') | (name=='yTestDV') | (name=='yTestDC2'):
            return self.decV[self.idxTest]
        # Lazy
        elif (name=='idxTrain') | (name=='idxTrainA') | (name=='idxTrainV') \
            | (name=='idxTrainC1') | (name=='idxTrainC2'):
            return self.trainIdx
        elif (name=='idxTest') | (name=='idxTestA') | (name=='idxTestV') \
            | (name=='idxTestC1') | (name=='idxTestC2'):
            return self.testIdx
        # Others
        else:
            return object.__getattribute__(self, name) 
        

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
                    