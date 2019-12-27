# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:53:14 2017

@author: Gareth
"""

#%% Imports
# Reload libraries that might have been edited
import importlib as il

from old import LSTMModels, utils, ConvModels

il.reload(LSTMModels)

il.reload(ConvModels)
from old.ConvModels import ConvModels

il.reload(utils)
from old.utils import dataHelpers as dh
from old.utils import multiChannelMod as AVMod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit as cFit
from scipy.special import erf

import seaborn as sns


#%% Load dataset
# s14 - AV
# s15 - A,V
# s16 - A, V, AV

dPath = 'Data/'
dSet = 'stimData_AV_s16_20000x400.mat'

# Load the dataset
data = dh(dPath+dSet, name='s16')
data = data.loadMatAV()

data = data.split(nTrain=10000)

data.plotDists()


#%% Model set up

# Save path
mPath = 'Models/'


#%% Load model to test

# Create appropriate object
name = 'AVConv_Late'
modAVConv = AVMod(mod=ConvModels(name=name).multiChanLate)

# Load from disk
modAVConv = modAVConv.load()

# Evaluate
# modAVConv = modAVConv.evaluate(data, setName='test')

# Predict
rateA, rateV, rateAV, decAV = modAVConv.predict([data.xTestExpAud, 
                                                 data.xTestExpVis])


#%% Prepare results

# MultiChanLate
# outputs=[rateOutput1, rateOutput2, rateOutputAV, decOutput]

# Mutichan early
# outputs=[rateOutput1, rateOutput2, rateOutputAV, decOutput]

results = pd.DataFrame({
                        'Type': data.testType.squeeze(),
                        'Rate' : np.max(np.concatenate((data.yTestRAud,data.yTestRVis), axis=1), axis=1), 
                        'estRateA' : rateA.squeeze(),
                        'estRateV' : rateV.squeeze(),
                        'estRateAV' : rateAV.squeeze(),
                        'estDecAV' : decAV[:,1].squeeze(),
                        'decA' : (rateA>np.mean(data.yTestRAud[data.testType==1])).squeeze(), 
                        'decV' : (rateV>np.mean(data.yTestRVis[data.testType==2])).squeeze(), 
                        'decAV' : np.round(decAV[:,1]).squeeze()
                        })

sns.regplot(results.Rate, results.estDecAV, x_jitter=1, y_jitter=0.1, 
            fit_reg=False)
plt.show()

fastProp = results.groupby(['Rate',  'Type']).mean()

sns.regplot(np.float32(fastProp.index), fastProp.estDecAV, fit_reg=False)
plt.show()
sns.regplot(np.float32(fastProp.index), fastProp.decAV, fit_reg=False)
plt.show()


#%% Curve fit

# WH fit
# F = @(g,l,u,v,x) g+(1-g-l)*0.5*(1+erf((x-u)/sqrt(2*v^2)));

def f(x, g=0.05, l=0.05, u=20, v=3):
    """
    WH function
    """
    return g+(1-g-l)*0.5*(1+erf((x-u)/np.sqrt(2*v**2)))


aIdx = fastProp.index.get_level_values(1)==1
subset = fastProp.iloc[aIdx]

popt, pcov = cFit(f, np.unique(results.Rate), fastProp.decAV[aIdx],
                  method='trf', p0=(0.01,0.01,11,1),
                  bounds=([0,0,0,1], [0.05, 0.05, 20., 3.]))

#p0 = (0.05,0.05,10,5),
x = np.array(range(np.min(data.ratesA)*100, np.max(data.ratesA)*100, 1))/100

y = f(x, g=popt[0], l=popt[1], u=popt[2], v=popt[3])

plt.scatter(np.unique(results.Rate), fastProp.decAV[aIdx])
plt.plot(x,y)
plt.show()


#%%

def fitPlot(xData, yData, lab = ''):
 
    popt, pcov = cFit(f, xData, yData,
                  bounds=([0,0,0,0], [0.05, 0.05, 20., 3.]))

    x = np.array(range(int(np.min(xData)*100), int(np.max(xData)*100), 1))/100
    y = f(x, g=popt[0], l=popt[1], u=popt[2], v=popt[3])

    plt.scatter(xData, yData)
    plt.plot(x,y, label=lab+'_fit')

typeIdx = fastProp.index.get_level_values(1)==3
fitPlot(np.unique(results.Rate), fastProp.decAV[typeIdx], lab='AV')
typeIdx = fastProp.index.get_level_values(1)==1
fitPlot(np.unique(results.Rate), fastProp.decA[typeIdx], lab='A')
typeIdx = fastProp.index.get_level_values(1)==2
fitPlot(np.unique(results.Rate), fastProp.decV[typeIdx], lab='V')
plt.legend()
plt.show()


# fitPlot(results.Rate, results.estDecAV, lab='AV')
fitPlot(results.Rate, results.decA, lab='A')
fitPlot(results.Rate, results.decV, lab='V')
plt.legend()

