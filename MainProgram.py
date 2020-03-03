"""
This Version: March 3, 2020. @copyright Hasan Fallahgoul, Vincentius Franstianto, and Gregoire Loeper 
If you use these codes, please cite the paper "Towards Explaining the ReLU Feed-Forward Networks: Asymptotic Properties" (2020) 
"""

#%%
import numpy as np
import numpy.random as npRnd
import numpy.matlib as mlb
import matplotlib.pyplot as plt
import scipy as sy
import scipy.interpolate as interpolate
import pandas as pd


from AuxilliaryFunctions import theSigmoid1
from AuxilliaryFunctions import theSigmoid2
from AuxilliaryFunctions import theSigmoid3
from AuxilliaryFunctions import theSigmoidSine1
from AuxilliaryFunctions import theSigmoidSine2
from AuxilliaryFunctions import theSigmoidSine3
from AuxilliaryFunctions import fluctuating
from AuxilliaryFunctions import nondiff
from AuxilliaryFunctions import feedForwardNetwork

#%% Main Program
    
npRnd.seed(1)

nSample = 50000

d = 1
x = npRnd.uniform(0.0,1.0,nSample)
epsilon = npRnd.normal(0.0,0.7,nSample)

if (not np.isin(1.0,x)):
    x = np.concatenate([x,[1.0]],axis=0)
    epsilon = np.concatenate([epsilon,npRnd.normal(0.0,0.7,1)],axis=0)
    pass

if (not np.isin(0.0,x)):
    x = np.concatenate([x,[0.0]],axis=0)
    epsilon = np.concatenate([epsilon,npRnd.normal(0.0,0.7,1)],axis=0)
    pass

print(np.isin(1,x))
print(np.isin(0,x))

part = 'part2'
thef0Name = 'nondiff'

if part=='part1':
    theFolder = 'Sigmoids and Its Variations'
if part=='part2':
    theFolder = 'fluctuating, Periodic, and Non-Differentiable'

#%%part1
if part=='part1':
    if (thef0Name == 'sigmoid1'):
        f0 = theSigmoid1(x)
        plottingName = 'Sigmoid 1'
    if (thef0Name == 'sigmoid2'):
        f0 = theSigmoid2(x)
        plottingName = 'Sigmoid 2'
    if (thef0Name == 'sigmoid3'):
        f0 = theSigmoid3(x)
        plottingName = 'Sigmoid'
    if (thef0Name == 'sigmoidSine1'):
        f0 = theSigmoidSine1(x)
        plottingName = 'Superposition'
    if (thef0Name == 'sigmoidSine2'):
        f0 = theSigmoidSine2(x)
        plottingName = 'Sigmoid Sine 2'
    if (thef0Name == 'sigmoidSine3'):
        f0 = theSigmoidSine3(x)
        plottingName = 'Sigmoid Sine 3'
    

#%%part2
if part=='part2':
    if (thef0Name == 'fluctuating'):
        f0 = fluctuating(x)
        plottingName = 'Periodic'
    if (thef0Name == 'nondiff'):
        [x,f0] = nondiff(x)
        plottingName = 'Non-Differentiable'

#%%
y = f0+epsilon

print('Sample Size = '+str(nSample))
print('')

activationFunction='sigmoid'
L = 1
H = int(np.rint(np.power(nSample,0.25)))
print('The Number of Nodes per Layer = ',H)
nHiddenUnitsFFN = []
removeProbFFN = []
removeProbFFN.append(0.0)
for i in range(1,L+1):
    nHiddenUnitsFFN.append(H)
    removeProbFFN.append(0.0)
    pass

batchSize = 32
epochSize = 32

theFFNSeed = 1
goTraining = True

if (activationFunction=='relu'):
    FFN = feedForwardNetwork(nHiddenUnitsFFN,removeProbFFN,x,y,\
                                    1,goTraining,theFolder+'/'+thef0Name+'/FFN for '+thef0Name+' with seed '+str(theFFNSeed)+','+str(L)+' layers, '+str(H)+' nodes per layer, '+str(nSample)+' sample data,'+str(batchSize)+' batches,'+str(epochSize)+' epochs.h5',\
                                    theFFNSeed,batchSize,epochSize,activationFunction)
    
if (activationFunction=='sigmoid'):
    FFN = feedForwardNetwork(nHiddenUnitsFFN,removeProbFFN,x,y,\
                                    1,goTraining,theFolder+'/'+thef0Name+'/SFN for '+thef0Name+' with seed '+str(theFFNSeed)+','+str(L)+' layers, '+str(H)+' nodes per layer, '+str(nSample)+' sample data,'+str(batchSize)+' batches,'+str(epochSize)+' epochs.h5',\
                                    theFFNSeed,batchSize,epochSize,activationFunction)

FFNValue = np.ndarray.flatten(FFN.predict(x))
rho0square = np.square(f0-FFNValue)
rho0square = np.average(rho0square)
mse = np.average(np.square(y-FFNValue))

#plt.figure()
#plt.plot(x,f0)
#plt.plot(np.arange(0.0,1.0+np.divide(1,x.shape[0]-1),np.divide(1,x.shape[0]-1)),FFNValue)

print('The rho square value = '+str(rho0square))
print('')
print('The mean squared error = '+str(mse))
print('')

x = np.reshape(x,(-1,1))

stepLength = 100000
xPlot = np.arange(0.0,1.0,1.0/stepLength)
f0Plot = interpolate.griddata(x,f0,xPlot,method='linear')
FFNValuePlot = interpolate.griddata(x,FFNValue,xPlot,method='linear')

plt.close('all')
plt.figure(1)
plt.plot(xPlot,f0Plot,'-b',label='True 'r'$f_{0}$')
if activationFunction=='relu':
    plt.plot(xPlot,FFNValuePlot,'-r',label='ReLU FFN, n='+str(nSample)+', L='+str(L)+', H='+str(H)+', batch='+str(batchSize)+', epoch='+str(epochSize))
if activationFunction=='sigmoid':
    plt.plot(xPlot,FFNValuePlot,'-m',label='Sigmoid FFN, n='+str(nSample)+', L='+str(L)+', H='+str(H)+', batch='+str(batchSize)+', epoch='+str(epochSize))
plt.legend(loc='lower right')
plt.title('Plot of The '+plottingName+' Function')
plt.xlabel('x')
plt.ylabel('value')
plt.show()



