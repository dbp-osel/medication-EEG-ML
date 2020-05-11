#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EEG-driven feature-based classifications and model generation
Software tools used generate results in submitted work to (tentative citation pending review):

David O. Nahmias, Eugene F. Civillico, and Kimberly L. Kontson. 
Deep Learning and Feature Based Medication Classifications from EEG in a Large Clinical Data Set 
In review (2020)


If you have found this software useful please consider citing our publication.

Public domain license
"""

""" Disclaimer:
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees
of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code,
this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge,
to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives,
and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other
parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied,
about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA
or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that
any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
"""

__author__ = 'David Nahmias'
__copyright__ = 'No copyright - US Government, 2020 , DeepEEG classification'
__credits__ = ['David Nahmias']
__license__ = 'Public domain'
__version__ = '0.0.1'
__maintainer__ = 'David Nahmias'
__email__ = 'david.nahmias@fda.hhs.gov'
__status__ = 'alpha'

import os
import numpy as np
import pdb
from collections import Counter
import scipy.stats as stats
from time import time
import datetime
import string
import sys
import pickle
#from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from metric_learn import LFDA
from multiprocessing import Pool
from itertools import combinations
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel,SelectKBest,f_classif,chi2
from sklearn.decomposition import PCA,NMF
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
import random

import nedcTools3

MODEL = 'linearModel'

def splitTrainTestData(allData,allLabels):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum
    #pdb.set_trace()
    sampleTrain1 = 0
    sampleTrain2 = 0
    sampleTest1 = 0
    sampleTest2 = 0
    allDataTrain = []
    allLabelsTrain = []
    allDataTest = []
    allLabelsTest = []
    for s in range(len(allLabels)):
        if (allLabels[s] == 0) and (sampleTrain1 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain1 += 1
        elif (allLabels[s] == 0) and (sampleTest1 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest1 += 1
        if (allLabels[s] == 1) and (sampleTrain2 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain2 += 1
        elif (allLabels[s] == 1) and (sampleTest2 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest2 += 1

    allDataTrain = np.array(allDataTrain)
    allLabelsTrain = np.array(allLabelsTrain)
    allDataTest = np.array(allDataTest)
    allLabelsTest = np.array(allLabelsTest)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest


def splitDataRotate(allData,allLabels,setNum=0):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = allData[allLabels == 0]
    data1 = allData[allLabels == 1]


    if setNum == -1 or setNum >= 10:
        testIndecies = range(trainSamplesNum,trainSamplesNum+testSamplesNum)
        trainIndecies = range(0,trainSamplesNum)

    elif (setNum < 10) and (setNum >= 0):
        testIndecies = range(setNum*testSamplesNum,(setNum+1)*testSamplesNum)
        trainIndecies = range(0,setNum*testSamplesNum) + range((setNum+1)*testSamplesNum,trainSamplesNum+testSamplesNum)

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest


def splitDataRandom(allData,allLabels,setNum=0,mode=[]):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = allData[allLabels == 0]
    data1 = allData[allLabels == 1]


    fullRange = range(numberEqSamples)
    random.shuffle(fullRange)

    np.save('%s/recordingsData/dataRangeOrder%s%s-recs'%(MODEL,mode[:4],mode[4]),fullRange)   #np.save('svmModel/dataRangeOrder%s-shuffle'%mode,fullRange)

    testIndecies = fullRange[trainSamplesNum:]
    trainIndecies = fullRange[:trainSamplesNum]

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)

    if mode[4] != '':
        random.shuffle(allLabelsTrain)
        np.save('%s/recordingsData/trainLabels%s%s-recs'%(MODEL,mode[:4],mode[4]),allLabelsTrain)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest

def splitDataRandom_Loaded(allData,allLabels,mode):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = allData[allLabels == 0]
    data1 = allData[allLabels == 1]


    #fullRange = list(range(numberEqSamples))
    #random.shuffle(fullRange)

    #np.save('dataRangeOrder%s'%CLASSY,fullRange)

    fullRange = np.load('%s/recordingsData/dataRangeOrder%s%s-recs.npy'%(MODEL,mode[:4],mode[4]))

    testIndecies = fullRange[trainSamplesNum:]
    trainIndecies = fullRange[:trainSamplesNum]

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=False) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    #pdb.set_trace()
    return ZCAMatrix

def normalizeFeatures(X):
    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0)
    return means,stds

def maxNormFeatures(X):
    means = np.mean(X,axis=0)
    maxi = np.max(np.abs(X),axis=0)
    return means,maxi   

def dataTransform(dataTrain,labelsTrain,dataTest,labelsTest,modes):
    for mode in modes:

        if 'normalize' in mode:
            featMeans,featStds = normalizeFeatures(dataTrain)
            dataTrain = np.divide(np.subtract(dataTrain,featMeans),np.multiply(featStds,3))
            dataTest = np.divide(np.subtract(dataTest,featMeans),np.multiply(featStds,3))

        if 'max' in mode:
            featMeans,featMax = maxNormFeatures(dataTrain)
            dataTrain = np.divide(np.subtract(dataTrain,featMeans),featMax)
            dataTest = np.divide(np.subtract(dataTest,featMeans),featMax)

        if 'pca' in mode:
            #U,S,V=np.linalg.svd(allDataTrain, full_matrices=0, compute_uv=1)
            #for i in range(1,134):
            #   pca = PCA(n_components=i)
            #   allDataTrainPCA = pca.fit_transform(allDataTrain)
            #   allDataEvalPCA = pca.fit_transform(allDataEval)
            
            pca = PCA(n_components=20)
            dataTrain = pca.fit_transform(dataTrain)
            dataTest = pca.fit_transform(dataTest)

        if 'lfda' in mode:
            #for i in range(1,134):
            #   ldfa = LFDA(k=7,dim=i)
            #   ldfa.fit(allDataTrain,allLabelsTrain)
            #   allDataTrainLFDA = ldfa.transform(allDataTrain)
            #   allDataEvalLFDA = ldfa.transform(allDataEval)

            ldfa = LFDA(k=10,dim=20)
            ldfa.fit(dataTrain,labelsTrain)
            dataTrain = ldfa.transform(dataTrain)
            dataTest = ldfa.transform(dataTest)

        if 'zca' in mode:
            zca = zca_whitening_matrix(dataTrain)
            dataTrain = np.dot(dataTrain,zca)
            dataTest = np.dot(dataTest,zca)

        
        if 'shuffle' in mode:
            random.shuffle(labelsTrain)

    return dataTrain,labelsTrain,dataTest,labelsTest

def mainSVM2(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)

    ##print('All:',len(allLabelsTest)+len(allLabelsTrain),'Train:',len(allLabelsTrain),'Test:',len(allLabelsTest))
    #return 0,0
    
    param_grid = [
        {'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
        ]
    #param_grid = [
    #   {'C':[100,1000],'gamma':[0.1],'kernel':['rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
    #   ]

    #selection = SelectKBest(f_classif,k=10)
    #selection.fit(allDataTrain,allLabelsTrain)

    #print(selection.get_support())

    #pdb.set_trace()

    svmAll = svm.SVC()#C=1000,gamma=0.001,kernel='rbf')
    clfAll = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)

    #clfAllpre = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)
    #clfAllpre.fit(allDataTrain,allLabelsTrain)
    
    #clfAll = Pipeline([
    #    ('feature_selection', SelectKBest(f_classif,k=10)),
    #    ('classification', svm.SVC(C=clfAllpre.best_params_['C'],kernel=clfAllpre.best_params_['kernel'],gamma=clfAllpre.best_params_['gamma']))])


    clfAll.fit(allDataTrain,allLabelsTrain)
    
    '''
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', None),
    ('classify', svm.SVC())
    ])

    N_FEATURES_OPTIONS = [6, 8, 10, 12]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(f_classif)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    ]
    reducer_labels = ['PCA', 'KBest(chi2)']

    clfAll = GridSearchCV(pipe, cv=10, n_jobs=20, param_grid=param_grid, iid=False, verbose=0)
    
    clfAll.fit(allDataTrain,allLabelsTrain)

    print(clfAll.best_params_)
    '''
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest)

def mainSVM(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad,mode=mode)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)

    ##print('All:',len(allLabelsTest)+len(allLabelsTrain),'Train:',len(allLabelsTrain),'Test:',len(allLabelsTest))
    #return 0,0
    
    '''
    param_grid = [
        {'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['linear','rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
        ]
    #param_grid = [
    #   {'C':[100,1000],'gamma':[0.1],'kernel':['rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
    #   ]

    selection = SelectKBest(f_classif,k=10)
    selection.fit(allDataTrain,allLabelsTrain)

    print(selection.get_support())

    pdb.set_trace()

    svmAll = svm.SVC()#C=1000,gamma=0.001,kernel='rbf')
    clfAll = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)

    #clfAllpre = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)
    #clfAllpre.fit(allDataTrain,allLabelsTrain)
    
    #clfAll = Pipeline([
    #    ('feature_selection', SelectKBest(f_classif,k=10)),
    #    ('classification', svm.SVC(C=clfAllpre.best_params_['C'],kernel=clfAllpre.best_params_['kernel'],gamma=clfAllpre.best_params_['gamma']))])


    clfAll.fit(allDataTrain,allLabelsTrain)
    '''
    
    
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', None),
    ('classify', svm.SVC())
    ])

    N_FEATURES_OPTIONS = [30, 60, 90, 120, 150]
    C_OPTIONS = [0.1, 1, 10, 100, 1000]
    GAMMA_OPTIONS = [0.1,0.01,0.001,0.0001,0.00001]
    KERNEL_OPTIONS = ['linear','rbf']
    param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(f_classif)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    ]
    reducer_labels = ['PCA', 'KBest(chi2)']

    clfAll = GridSearchCV(pipe, cv=10, n_jobs=30, param_grid=param_grid, iid=False, verbose=0)

    clfAll.fit(allDataTrain,allLabelsTrain)
    
    print(clfAll.best_params_)
    
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest),clfAll

def evalSVMall(clf,allLabelsEval,allDataEval,Write2File=0):
    #Write2File = 1

    correctPredict = 0
    
    t1correctPredict = 0
    t2correctPredict = 0
    
    for ind in range(np.shape(allDataEval)[0]):
        curPredict = clf.predict(allDataEval[ind].reshape(1,-1))[0]
        if (curPredict==allLabelsEval[ind]):
            correctPredict += 1
            if curPredict == 0:
                t1correctPredict += 1
            else:
                t2correctPredict += 1
    if Write2File == 0:
        #print 'Channels:',ch
        print 'Correct Predictions, overall:',correctPredict,'out of',len(allLabelsEval),'(',100.*correctPredict/len(allLabelsEval),')'
        print 'Correct Predictions, Class 1:',t1correctPredict,'out of',len(allLabelsEval) - np.sum(allLabelsEval)
        print 'Correct Predictions, Class 2:',t2correctPredict,'out of', np.sum(allLabelsEval),'\n'
    return 100.*correctPredict/len(allLabelsEval)

def mainRBF(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad,mode=mode)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)

    ##print('All:',len(allLabelsTest)+len(allLabelsTrain),'Train:',len(allLabelsTrain),'Test:',len(allLabelsTest))
    #return 0,0
    
    '''
    param_grid = [
        {'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['linear','rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
        ]
    #param_grid = [
    #   {'C':[100,1000],'gamma':[0.1],'kernel':['rbf']},#{'C':[0.1,1,10,100,1000],'gamma':[0.1,0.01,0.001,0.0001,0.00001],'kernel':['rbf']},
    #   ]

    selection = SelectKBest(f_classif,k=10)
    selection.fit(allDataTrain,allLabelsTrain)

    print(selection.get_support())

    pdb.set_trace()

    svmAll = svm.SVC()#C=1000,gamma=0.001,kernel='rbf')
    clfAll = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)

    #clfAllpre = GridSearchCV(svmAll,param_grid,n_jobs=20,verbose=0)
    #clfAllpre.fit(allDataTrain,allLabelsTrain)
    
    #clfAll = Pipeline([
    #    ('feature_selection', SelectKBest(f_classif,k=10)),
    #    ('classification', svm.SVC(C=clfAllpre.best_params_['C'],kernel=clfAllpre.best_params_['kernel'],gamma=clfAllpre.best_params_['gamma']))])


    clfAll.fit(allDataTrain,allLabelsTrain)
    '''
    
    
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', None),
    ('classify', svm.SVC())
    ])

    N_FEATURES_OPTIONS = [60, 120, 210, 390, 570]
    C_OPTIONS = [1, 10, 100, 1000]
    GAMMA_OPTIONS = [0.1,0.01,0.001]
    KERNEL_OPTIONS = ['rbf']
    param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(f_classif)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    ]
    reducer_labels = ['PCA', 'KBest(chi2)']

    clfAll = GridSearchCV(pipe, cv=10, n_jobs=30, param_grid=param_grid, iid=False, verbose=0)

    clfAll.fit(allDataTrain,allLabelsTrain)
    
    print(clfAll.best_params_)
    
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest),clfAll



def mainLinear(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad,mode=mode)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)
    
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', None),
    ('classify', svm.SVC())
    ])

    N_FEATURES_OPTIONS = [60, 120, 210, 390, 570]
    C_OPTIONS = [1, 10, 100, 1000]
    KERNEL_OPTIONS = ['linear']
    param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(f_classif)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS
    },
    ]
    reducer_labels = ['PCA', 'KBest(chi2)']


    clfAll = GridSearchCV(pipe, cv=10, n_jobs=30, param_grid=param_grid, iid=False, verbose=0)

    clfAll.fit(allDataTrain,allLabelsTrain)
    
    print(clfAll.best_params_)
    
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest),clfAll

def mainRFGrid(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad,mode=mode)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)
    
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('classify', RandomForestClassifier())
    ])

    MAX_DEPTH = [10, 20, None]
    N_ESTIMATORS = [50, 100, 200]
    MIN_LEAF = [1, 2, 4]
    MIN_SPLIT = [2, 4, 8]
    RANDOM_STATE = [0]
    param_grid = [
    {
        'max_depth': MAX_DEPTH,
        'n_estimators': N_ESTIMATORS,
        'min_samples_leaf': MIN_LEAF,
        'min_samples_split': MIN_SPLIT,
        'random_state': RANDOM_STATE
    }
    ]

    clfAll = GridSearchCV(pipe, cv=10, n_jobs=30, param_grid=param_grid, iid=False, verbose=0)

    #clfAll = RandomForestClassifier(n_estimators=100, n_jobs=30, random_state=0, verbose=0)

    clfAll.fit(allDataTrain,allLabelsTrain)
    
    #print(clfAll.get_params)
    
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest),clfAll

def mainRF(dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):

    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitTrainTestData(dataLoad,labelsLoad)
    
    #allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRotate(dataLoad,labelsLoad,setNum=setNum)
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom(dataLoad,labelsLoad,mode=mode)


    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)


    clfAll = RandomForestClassifier(n_estimators=100, n_jobs=30, random_state=0, verbose=0)

    clfAll.fit(allDataTrain,allLabelsTrain)
    
    #print(clfAll.get_params)
    
    #pdb.set_trace()

    ##print('AUC',metrics.roc_auc_score(allLabelsTest,clfAll.predict(allDataTest)))
    #print 'SVM:',clfNorm
    
    ##print(clfAll.best_params_)

    if evalSet == 1:
        halfPoint = int(np.floor(len(allLabelsTest)))
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest[0:halfPoint],allDataTest[0:halfPoint]),evalSVMall(clfAll,allLabelsTest[halfPoint:],allDataTest[halfPoint:])
    else:
        return evalSVMall(clfAll,allLabelsTrain,allDataTrain),evalSVMall(clfAll,allLabelsTest,allDataTest),clfAll


def remove_punctuation(value):
    result = ''
    for c in value:
        # If char is not punctuation, add it to the result.
        if c not in string.punctuation:
            result += c
    return result

def getSubjArray(labels):
    subjArray = []
    for i in labels[:,0]:
        subjArray.append(i.split('_')[0])
    return np.array(subjArray)

def getSessionArray(labels):
    sessArray = []
    for i in labels[:,0]:
        sessArray.append(i.split('_')[0]+'_'+i.split('_')[1])
    return np.array(sessArray)

def getDiffSessions(labels,subjArray,subjCaptured,subj):
    curSubjSess = []
    ind = np.where(subjArray == subj)[0]
    #pdb.set_trace()

    if len(ind)>1:
        for i in range(len(ind)-1):
            sessionNameCur = labels[ind[i]][0].split('_')
            sessionNameNext = labels[ind[i+1]][0].split('_')
            if (sessionNameCur[0] == sessionNameNext[0]) and (sessionNameCur[1] != sessionNameNext[1]):
                #pdb.set_trace()

                if subj not in subjCaptured:
                    #pdb.set_trace()
                    subjCaptured.append(subj)
                curSubjSess.append(ind[i])
                if (i == len(ind)-2):
                    curSubjSess.append(ind[i+1])
            
            elif (i>0):
                if (sessionNameCur[0] == labels[ind[i-1]][0].split('_')[0]) and (sessionNameCur[1] != labels[ind[i-1]][0].split('_')[1]):
                    curSubjSess.append(ind[i])

    return subjCaptured,curSubjSess

def getDateDiff(dates):
    #print 'Dates:',dates
    dateDiff = max(dates)-min(dates)
    #print 'Diff:',dateDiff, 'Days:', dateDiff.days
    return dateDiff.days

def getMeanAge(ages):

    if (len(ages[0]) + len(ages[1])) > 1:
        meanAge = np.mean([ages[0][0],ages[1][0]])
    elif len(ages[0]) > 0:
        meanAge = ages[0][0]
    elif len(ages[1]) > 0:
        meanAge = ages[1][0]
    else:
        meanAge = -1

    return meanAge

def dataSummary(labels):
    #allLabels.append([dataName,dateCur,val.subjGender,val.age,getMedsListStr(val.subjMed),val.subjNormalState])

    allNormal = 0
    ages = []
    normal = 0
    abnormal = 0
    noNA = 0
    male = 0
    female = 0
    noSex = 0
    
    subjArray = getSubjArray(labels)
    
    subjCaptured = []
    subjCapturedS = []
    subjCapturedA = []

    print('Total Sessions:',len(subjArray))
    for i in range(len(subjArray)):
        curSubj = subjArray[i]

        if curSubj not in subjCaptured:
            subjCaptured.append(curSubj)
        
        #if curSubj not in subjCapturedS:
        if labels[i][2] == 'male':
            male += 1
            subjCapturedS.append(curSubj)        
        if labels[i][2] == 'female':
            female += 1
            subjCapturedS.append(curSubj)        
        if (labels[i][2] != 'male') and (labels[i][2] != 'female'):
            noSex += 1
    
        #if curSubj not in subjCapturedA:
        if len(labels[i][3])>0:
            ages.append(labels[i][3][0])
            subjCapturedA.append(curSubj)        


        if labels[i][5] == 0:
            normal += 1
        if labels[i][5] == 1:
            abnormal += 1
        if labels[i][5] == 2:
            noNA += 1

    print('Males:',male,'; Female:',female,'; Neither:',noSex,' Total:',male+female+noSex)
    print('Age: Mean:',np.mean(ages),' SD:',np.std(ages),' IQR:',stats.iqr(ages),' n=',len(ages))
    print('Normal:',normal,'; Abnormal:',abnormal,'; Neither:',noNA,' Total:',normal+abnormal+noNA)
    print('Unique Subjects Found:',len(np.unique(subjArray)),'\n')
    
    subjCaptured = []
    multiSession = 0
    allDates = []
    allAges = []
    subj1 = 0
    subj2 = 1
    useRand = 0
    for i in range(len(labels)):
        curSubj = labels[i][0].split('_')[0]
        if curSubj in subjCaptured:
            continue
        subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)


        if len(dataCaptured) > 1:
            if useRand == 1:
                randSubj = -1
                while (randSubj in subjCaptured) or (randSubj<0):
                    randSubj = random.randint(0,len(labels)-1) 
                dataCaptured[subj2] = randSubj

            multiSession += 1
            allAges.append(getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]]))
            allDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))
    allAges = np.array(allAges)
    allDates = np.array(allDates)

    print('Number of Multi-Session Subjects:',multiSession)
    print('Time between first and second visit (days): Median:',np.median(allDates),' SD:',np.std(allDates),' IQR:',stats.iqr(allDates),' n=',len(allDates))
    print('Mean Ages of subjects with multiple sessions: Mean:',np.mean(allAges[allAges>0]),' SD:',np.std(allAges[allAges>0]),' IQR:',stats.iqr(allAges[allAges>0]),' n=',np.size(allAges[allAges>0]))


def convertKeyWordsToStr(KWarray):
    KWarrayStr = []
    for w in KWarray:
        try:
            curKW = str(w.decode('UTF-8'))
        except UnicodeDecodeError:
            curKW = ''
        
        KWarrayStr.append(curKW)

    return KWarrayStr

def convertLabelsToString(labels):
    newLabels = []
    for l in labels:
        try:
            medWords = str(l[4].decode('UTF-8'))
        except UnicodeDecodeError:
            medWords = ''
        except UnicodeEncodeError:
            medWords = ''
        curLabel = [str(l[0].decode('UTF-8')),l[1],str(l[2].decode('UTF-8')),l[3],medWords,l[5],convertKeyWordsToStr(l[6])]

        newLabels.append(curLabel)

    return np.array(newLabels)

def compareTwoMed(medLabels,normality,norm=0):
    badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
        'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
        'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
        'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran',
        'dilantin','ativan','depakote']
    
    top5 = ['dilantin','keppra','depakote','tegretol','lamictal']
    permuteTop5 = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

    for p in permuteTop5:
    
        allLabels = []
        allMeds = []
        for c in range(len(medLabels)):
            curMedArray = medLabels[c].split()
            curNormal = normality[c]
            if len(curMedArray) != 2:
                continue

            medList = []
            for w in curMedArray:
                w = remove_punctuation(w.lower())
                if (w in badMeds):# and (curNormal == 0):
                    if (top5[p[0]] in curMedArray) and (top5[p[1]] in curMedArray):
                        medList.append(w)
            
            allLabels.append(medList)
            allMeds.extend(medList)

        allLabels = np.array(allLabels)

        unique, counts = np.unique(allMeds, return_counts=True)

        totCounts = Counter(dict(zip(unique, counts)))

        print(str(totCounts.most_common(50)))

def singleMed(medLabels,normality,normMeds=0,norm=1):
    badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
        'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
        'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
        'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran',
        'dilantin','ativan','depakote']
    
    top5 = ['dilantin','keppra','depakote','tegretol','lamictal']
    permuteTop5 = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

    
    allLabels = []
    allMeds = []
    subjectIndex = []
    for c in range(len(medLabels)):
        curMedArray = medLabels[c].split()
        curNormal = normality[c]
        if len(curMedArray) > 1:
            continue

        medList = []
        for w in curMedArray:
            w = remove_punctuation(w.lower())
            if normMeds == 0:
                if (w in badMeds) and (curNormal == norm):
                    medList.append(w)
                    subjectIndex.append(c)
            elif normMeds == 1:
                if (w not in badMeds) and (curNormal == norm):
                    medList.append(w)
                    subjectIndex.append(c)

        allLabels.append(medList)
        allMeds.extend(medList)

    allLabels = np.array(allLabels)

    unique, counts = np.unique(allMeds, return_counts=True)

    totCounts = Counter(dict(zip(unique, counts)))

    print(str(totCounts.most_common(50)))


    return allLabels,allMeds


def mainLabels(data,labels,interestMed = ['dilantin','keppra'],norm=0):
    '''
    PATH = '/media/david/Data2/scattering/results/'

    labels_path = PATH + 'scatteringLabels_Q8_J8_T5m_all'
    labels1 = np.load(labels_path+'.npy')
	'''

    #PATH = '/media/david/Data2/spectralGraph/results/'
    #labels_path = PATH + 'coherenceLabels_all'
    #labels = np.load(labels_path+'.npy',encoding='bytes')

    labels = convertLabelsToString(labels)

    #dataSummary(labels)
    subjectNames = np.unique(getSubjArray(labels))
    sessionNames = getSessionArray(labels)
    uniqueSessionNames = np.unique(sessionNames)
    sessionMask = []#np.zeros(labels[:,0].shape).astype('int')
    
    for s in uniqueSessionNames:
        sessionMask.append(np.where(s == sessionNames)[0][0])
        #sessionMask[ind] = 1

    dataSessions = data[sessionMask]
    #pdb.set_trace()
    medLabels = labels[sessionMask,4]
    normality = labels[sessionMask,5]

    dataSessions = data
    medLabels = labels[:,4]
    normality = labels[:,5]
    #pdb.set_trace()

    #compareTwoMed(medLabels,normality,0)
    #compareTwoMed(medLabels,normality,1)

    #singleMed(medLabels,normality,0,0)
    #singleMed(medLabels,normality,0,1)
    #singleMed(medLabels,normality,1,0)
    #singleMed(medLabels,normality,1,1)

    badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
        'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
        'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
        'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran',
        'dilantin','ativan','depakote']
    
    top5 = ['dilantin','keppra','depakote','tegretol','lamictal']
    permuteTop5 = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

    allLabels = []
    allMeds = []
    subjectIndex = []
    med1Index = []
    med2Index = []

    '''
    for c in range(len(normality)):
        curNormal = normality[c]

        if (curNormal == 0):
            med1Index.append(c)
            subjectIndex.append(c)
            allLabels.append(0)
        elif (curNormal == 1):
            med2Index.append(c)
            subjectIndex.append(c)
            allLabels.append(1)

    allLabels = np.array(allLabels)
    dataLabeled = dataSessions[subjectIndex]

    selectedLabels = labels[sessionMask]
    dataSummary(selectedLabels)

    selectedAllLabels = selectedLabels[med1Index]
    dataSummary(selectedAllLabels)

    selectedAllLabels = selectedLabels[med2Index]
    dataSummary(selectedAllLabels)
    '''

    for c in range(len(medLabels)):
        curMedArray = medLabels[c].split()
        curNormal = normality[c]
        if len(curMedArray) > 1:
            continue

        for w in curMedArray:
            w = remove_punctuation(w.lower())
            if (w == interestMed[0]) and (curNormal == norm):
                med1Index.append(c)
                subjectIndex.append(c)
                allLabels.append(0)
            elif (w == interestMed[1]) and (curNormal == norm):
                med2Index.append(c)
                subjectIndex.append(c)
                allLabels.append(1)
        
    allLabels = np.array(allLabels)
    dataLabeled = dataSessions[subjectIndex]

    '''
    selectedLabels = labels[sessionMask]
    dataSummary(selectedLabels)

    selectedAllLabels = selectedLabels[med1Index]
    dataSummary(selectedAllLabels)

    selectedAllLabels = selectedLabels[med2Index]
    dataSummary(selectedAllLabels)

    selectedAllLabels = selectedLabels[subjectIndex]
    dataSummary(selectedAllLabels)
    '''
    #pdb.set_trace()
    
    return dataLabeled,allLabels

def mainFeatureCompute(filePath):

    path = filePath #processed Data
    print('Loading: %s'%(filePath))

    allData = []
    allLabels = []
    f = open(path,'rU')
    content = f.readlines()

    features = 'all'
    bands = [1,4,8,12,16,25,40]
    srate = 100
    partitions = 1
    
    for file in content:
        print('Loading',file.strip())

        curLoad = np.load('/media/david/Data1/'+file.strip(),encoding='bytes',allow_pickle=True)
        
        curLabel = curLoad[0]
        curData = np.float32(curLoad[1])


        dataFeatures = nedcTools3.getAllFeatures(curData,features,bands,srate,partitions)
        
        #pdb.set_trace()

        allData.append(dataFeatures)
        allLabels.append(curLabel)

    return np.array(allData),np.array(allLabels)


def evalLoadedSVM(clfAll,dataLoad,labelsLoad,transform = '',evalSet=0,setNum=-1,mode=[]):
    
    allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = splitDataRandom_Loaded(dataLoad,labelsLoad,mode=mode)

    if len(transform)>0: 
        allDataTrain,allLabelsTrain,allDataTest,allLabelsTest = dataTransform(allDataTrain,allLabelsTrain,allDataTest,allLabelsTest,transform)

    return evalSVMall(clfAll,allLabelsTrain,allDataTrain,Write2File=1),evalSVMall(clfAll,allLabelsTest,allDataTest,Write2File=1)

def testSVMResults():
    transforms = ['max']
    numIter = 10

    #meanResults = np.zeros((numIter,1))
    #trainRes = np.zeros((numIter,1))

    #meanResultsShuffle = np.zeros((numIter,1))
    #trainResShuffle = np.zeros((numIter,1))


    data = np.load(data_path+'.npy',encoding='bytes',allow_pickle=True)
    labels = np.load(labels_path+'.npy',encoding='bytes',allow_pickle=True)
    
    #pdb.set_trace()

    allResults = []

    allC = []
    allGamma = []
    allDim = []
    allMeth = []

    classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','keppra',0],['none','dilantin',1],['none','keppra',1]]
    classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]

    for mode in classes:
        
        labeledData,labeledLabels = mainLabels(data,labels,interestMed = mode[0:2],norm=mode[2])
        labeledData = np.reshape(labeledData,(labeledData.shape[0],labeledData.shape[1]*labeledData.shape[2]))

        meanResults = []
        trainRes = []
        meanResultsShuffle = []
        trainResShuffle = []

        for i in range(numIter):
            modeNameList = [mode[0],mode[1],mode[2],i,'']
            modeNameListStr = [str(mode[0]),str(mode[1]),str(mode[2]),str(i),modeNameList[4]]
            modeName = '-'.join(modeNameListStr[:4])
            filenameModel = '%s/recordingsData/%sModel%s-recs.mod'%(MODEL,modeName,modeNameListStr[4])
            clfAll = pickle.load(open(filenameModel, 'rb'))
            #print('Parameters:',clfAll.best_params_)
            #print('Parameters:',clfAll.get_params)
            
            allC.append(clfAll.best_params_['classify__C'])
            allMeth.append(str(clfAll.best_params_['reduce_dim'])[0])
            if str(clfAll.best_params_['reduce_dim'])[0] == 'P':
                allDim.append(clfAll.best_params_['reduce_dim__n_components'])
            elif str(clfAll.best_params_['reduce_dim'])[0] == 'S':
                allDim.append(clfAll.best_params_['reduce_dim__k'])
            try:
                allGamma.append(clfAll.best_params_['classify__gamma'])
            except:
                allGamma.append(0)
            results = evalLoadedSVM(clfAll,labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)
            
            #meanResults[i] = results[1]
            #trainRes[i] = results[0]
            
            meanResults.append(results[1])
            trainRes.append(results[0])

            modeNameList = [mode[0],mode[1],mode[2],i,'shuffle']
            modeNameListStr = [str(mode[0]),str(mode[1]),str(mode[2]),str(i),modeNameList[4]]
            modeName = '-'.join(modeNameListStr[:4])
            filenameModel = '%s/recordingsData/%sModel%s-recs.mod'%(MODEL,modeName,modeNameListStr[4])
            clfAll = pickle.load(open(filenameModel, 'rb'))

            results = evalLoadedSVM(clfAll,labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)
            
            #meanResultsShuffle[i] = results[1]
            #trainResShuffle[i] = results[0]

            meanResultsShuffle.append(results[1])
            trainResShuffle.append(results[0])

        uniqueC, countsC = np.unique(allC[-numIter:], return_counts=True)
        print("Local C:",uniqueC,countsC)
    
        uniqueG, countsG = np.unique(allGamma[-numIter:], return_counts=True)
        print("Local Gamma:",uniqueG,countsG)

        uniqueM, countsM = np.unique(allMeth[-numIter:], return_counts=True)
        print("Local Method:",uniqueM,countsM)

        uniqueD, countsD = np.unique(allDim[-numIter:], return_counts=True)
        print("Local Dimensions:",uniqueD,countsD)
    

        allResults.append(meanResults)
        F,p = stats.kruskal(meanResults,meanResultsShuffle) 

        print("Train Mean: interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(trainRes),np.std(trainRes)))
        print("Test Mean: interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(meanResults),np.std(meanResults)))

        print("Train Mean (Random): interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(trainResShuffle),np.std(trainResShuffle)))
        print("Test Mean (Random): interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(meanResultsShuffle),np.std(meanResultsShuffle)))

        print('Significance: interestMed = %s, norm=%d - p =%f'%(mode[0:2],mode[2],p))

    uniqueC, countsC = np.unique(allC, return_counts=True)
    print("C:",uniqueC,countsC)
    
    uniqueG, countsG = np.unique(allGamma, return_counts=True)
    print("Gamma:",uniqueG,countsG)

    uniqueM, countsM = np.unique(allMeth, return_counts=True)
    print("Method:",uniqueM,countsM)

    uniqueD, countsD = np.unique(allDim, return_counts=True)
    print("Dimensions:",uniqueD,countsD)

    toCompare = [[0,2],[0,3],[1,4],[1,5],[2,3],[4,5]]
    toCompare = [[0,2],[0,4],[1,3],[1,5],[2,4],[3,5]]

    for c in toCompare:
        #pdb.set_trace()
        F,p = stats.kruskal(allResults[c[0]],allResults[c[1]]) 

        print('Significance: %s v. %s - p =%f'%(classes[c[0]],classes[c[1]],p))

    np.save('linear-recs-results',allResults)


if __name__ == '__main__':
    startGlobal = time()

    computeFeatures = 0
    evalFeatures = 0
    testResults = 1


    PATH = '/media/david/Data2/featuresComputed/results/'

    data_path = PATH + 'featuresData_5m_all'
    labels_path = PATH + 'featuresLabels_5m_all'

    if computeFeatures == 1:
        INPUTFILENAME = str(sys.argv[1])#str(sys.argv[1]).split('.')[0]
        data,labels = mainFeatureCompute(filePath=INPUTFILENAME)
        np.save(data_path, data )
        np.save(labels_path, labels )

        endGlobal = time()
        print('Time elapsed for all features: %s'%(str(endGlobal-startGlobal)))

    if evalFeatures == 1:
    
        transforms = ['max']
        numIter = 10

        meanResults = np.zeros((numIter,1))
        trainRes = np.zeros((numIter,1))

        data = np.load(data_path+'.npy',encoding='bytes',allow_pickle=True)
        labels = np.load(labels_path+'.npy',encoding='bytes',allow_pickle=True)
        
        #pdb.set_trace()
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]

        for mode in classes:
            
            labeledData,labeledLabels = mainLabels(data,labels,interestMed = mode[0:2],norm=mode[2])
            labeledData = np.reshape(labeledData,(labeledData.shape[0],labeledData.shape[1]*labeledData.shape[2]))
            #print('len:',labeledLabels.shape,'1s:',sum(labeledLabels))
            #continue
            for i in range(numIter):
                modeNameList = [mode[0],mode[1],mode[2],i,'shuffle']#'shuffle'
                #results = mainSVM(labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)
                results = mainLinear(labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)
                #results = mainRBF(labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)
                #results = mainRFGrid(labeledData,labeledLabels,transform=transforms,setNum=i,mode=modeNameList)

                meanResults[i] = results[1]
                trainRes[i] = results[0]
                
                modeNameList = [str(mode[0]),str(mode[1]),str(mode[2]),str(i),modeNameList[4]]
                modeName = '-'.join(modeNameList[:4])
                #filenameModel = 'svmModel/recordingsData/%sModel%s-recs.mod'%(modeName,modeNameList[4])
                filenameModel = '%s/recordingsData/%sModel%s-recs.mod'%(MODEL,modeName,modeNameList[4])
                #filenameModel = 'rbfModel/recordingsData/%sModel%s-recs.mod'%(modeName,modeNameList[4])
                #filenameModel = 'rfGridModel/recordingsData/%sModel%s-recs.mod'%(modeName,modeNameList[4])

                pickle.dump(results[2], open(filenameModel, 'wb'))
            
            print("Test Mean: interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(meanResults),np.std(meanResults)))
            print("Train Mean: interestMed = %s, norm=%d - %f (%f) \n"%(mode[0:2],mode[2],np.mean(trainRes),np.std(trainRes)))

        endGlobal = time()
        print('Time elapsed for all features: %s'%(str(endGlobal-startGlobal)))

    if testResults == 1:
        testSVMResults()

        endGlobal = time()
        print('Time elapsed for all tests: %s'%(str(endGlobal-startGlobal)))

