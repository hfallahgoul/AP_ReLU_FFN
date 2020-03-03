
"""
This Version: March 3, 2020. @copyright Hasan Fallahgoul, Vincentius Franstianto, and Gregoire Loeper 
If you use these codes, please cite the paper "Towards Explaining the ReLU Feed-Forward Networks: Asymptotic Properties" (2020) 
"""

#%%
import numpy as np
import numpy.random as npRnd
import numpy.matlib as mlb

import tensorflow as tf
import keras as krs
import keras.layers as kLayers
import keras.backend as kB
#%% FFN

def feedForwardNetwork (nHiddenUnits,removeProb,xData,yData,nInput,goTraining,filename,theSeed,batchSize,epochSize,activationFunction):
    if goTraining:
        tf.reset_default_graph()
        config1 = tf.ConfigProto()
        config1.intra_op_parallelism_threads = 4
        config1.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config1)
        kB.set_session(sess)
    
    theModel = krs.models.Sequential()
    theModel.add(kLayers.InputLayer(input_shape=(nInput,)))
    theModel.add(kLayers.Dropout(rate=removeProb[0]))
    if nHiddenUnits:
        for i in range(0,len(nHiddenUnits)):
            theModel.add(kLayers.Dense(nHiddenUnits[i],activation=activationFunction,use_bias=True,\
                                       kernel_initializer=krs.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=theSeed),\
                                       bias_initializer=krs.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=theSeed)))
            theModel.add(kLayers.Dropout(rate=removeProb[i+1],seed=theSeed))
    theModel.add(kLayers.Dense(1,use_bias=True,kernel_initializer=krs.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=theSeed),\
                                           bias_initializer=krs.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=theSeed)))
    
    if goTraining:
        theOptimizer = krs.optimizers.Nadam(lr=0.001)
        theModel.compile(loss='mean_squared_error',optimizer=theOptimizer,metrics=['mean_squared_error'])
        theModel.fit(x=xData,y=yData,batch_size=batchSize,epochs=epochSize)
        theModel.save_weights(filename)
    else:
        theModel.load_weights(filename)
    return theModel

#%% Sigmoid
    
def theSigmoid1(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return -1 + 5.0*sigm1 - 3.0*sigm2
#%% Sigmoid 2
    
def theSigmoid2(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return  8.0*sigm1 - 2.0*sigm2 + 5.0
#%% Sigmoid 3
    
def theSigmoid3(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return  18.0*sigm1 - 12.0*sigm2 + 5.0

#%% Sigmoid Sine
    
def theSigmoidSine1(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return  18.0*sigm1 - 12.0*sigm2 + 5.0*np.sin(8.0*np.pi*x)
#%% Sigmoid Sine 2
    
def theSigmoidSine2(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return  18.0*sigm1 - 12.0*sigm2 + 10.0*np.sin(16.0*np.pi*x)
#%% Sigmoid Sine 3
    
def theSigmoidSine3(x):
    sigm1 = np.exp(9.0*x-2.0)
    sigm1 = np.divide(sigm1,1.0+sigm1)
    
    sigm2 = np.exp(2.0-9.0*x)
    sigm2 = np.divide(sigm2,1.0+sigm2)
    
    
    return  -18.0*sigm1 + 12.0*sigm2 + 10.0*np.sin(16.0*np.pi*x)
#%% fluctuating
    
def fluctuating(x):
    return  np.sin(2.0*np.pi*x) + (1/3)*np.cos(3.0*np.pi*x+3)

#%% Nondiff
    
def nondiff(x):
    lHalf = np.argwhere(x<=0.5)
    gHalf = np.argwhere(x>0.5)
    
    lHalf = np.reshape(lHalf,(-1,))
    gHalf = np.reshape(gHalf,(-1,))
    
    firstHalf = -8.0*(x[lHalf]-0.5)
    secondHalf = 10.0*np.sqrt(x[gHalf]-0.5)*(2.0 - x[gHalf])
    
    newX = np.concatenate([x[lHalf],x[gHalf]],axis=0)
    f0 = np.concatenate([firstHalf,secondHalf],axis=0)
    
    return [newX,f0]
