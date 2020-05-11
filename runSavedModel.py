#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Display results from EEG-driven deep learning classifications and model generation and plots from publication
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
import logging
import time
from copy import copy
import sys

from scipy import stats
from collections import Counter
import random
import numpy as np
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from collections import OrderedDict
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import pdb
from loadNEDC import loadNEDCdata,loadSubNormData,addDataNoise

from auto_diagnosis import create_set,TrainValidTestSplitter,TrainValidSplitter,run_exp

import config

import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

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

    #fullRange = np.load('sessionsData/dataRangeOrder%s%s-sessions.npy'%(mode[:4],mode[4]))
    fullRange = np.load('linear/dataRangeOrder%s%s.npy'%(mode[:4],mode[4]))


    testIndecies = fullRange[trainSamplesNum:]
    trainIndecies = fullRange[:trainSamplesNum]

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest

def runModel(mode):
    cudnn.benchmark = True

    start = time.time()


    #mode = str(sys.argv[1])
    #X,y,test_X,test_y = loadSubNormData(mode='all')
    #X,y,test_X,test_y = loadNEDCdata(mode=mode)

    #data = np.load('sessionsData/data%s-sessions.npy'%mode[:3])
    #labels = np.load('sessionsData/labels%s-sessions.npy'%mode[:3])

    data = np.load('data%s.npy'%mode[:3])
    labels = np.load('labels%s.npy'%mode[:3])

    X,y,test_X,test_y = splitDataRandom_Loaded(data,labels,mode)

    print('Mode - %s Total n: %d, Test n: %d'%(mode,len(y)+len(test_y),len(test_y)))
    #return 0


    #X = addDataNoise(X,band=[1,4])
    #test_X = addDataNoise(test_X,band=[1,4])

    max_shape = np.max([list(x.shape) for x in X],axis=0)

    assert max_shape[1] == int(config.duration_recording_mins * config.sampling_freq * 60)

    n_classes = 2
    n_recordings = None  # set to an integer, if you want to restrict the set size
    sensor_types = ["EEG"]
    n_chans = 19#21
    max_recording_mins = 35  # exclude larger recordings from training set
    sec_to_cut = 60  # cut away at start of each recording
    duration_recording_mins = 5#20  # how many minutes to use per recording
    test_recording_mins = 5#20
    max_abs_val = 800  # for clipping
    sampling_freq = 100
    divisor = 10  # divide signal by this
    test_on_eval = True  # teston evaluation set or on training set
    # in case of test on eval, n_folds and i_testfold determine
    # validation fold in training set for training until first stop
    n_folds = 10
    i_test_fold = 9
    shuffle = True
    model_name = 'linear'#'deep'#'shallow' 'linear'
    n_start_chans = 25
    n_chan_factor = 2  # relevant for deep model only
    input_time_length = 6000
    final_conv_length = 1
    model_constraint = 'defaultnorm'
    init_lr = 1e-3
    batch_size = 64
    max_epochs = 35 # until first stop, the continue train on train+valid
    cuda = True # False

    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2 = int(n_start_chans * n_chan_factor),
                         n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                        stride_before_pool=True).create_network()
    elif (model_name == 'deep_smac'):
        if model_name == 'deep_smac':
            do_batch_norm = False
        else:
            assert model_name == 'deep_smac_bnorm'
            do_batch_norm = True
        double_time_convs = False
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 12
        filter_time_length = 21
        final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = 'mean'
        later_pool_nonlin = identity
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(n_chans, n_classes,
                 n_filters_time=n_start_chans,
                 n_filters_spat=n_start_chans,
                 input_time_length=input_time_length,
                 n_filters_2=int(n_start_chans * n_chan_factor),
                 n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                 n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                 final_conv_length=final_conv_length,
                 batch_norm=do_batch_norm,
                 double_time_convs=double_time_convs,
                 drop_prob=drop_prob,
                 filter_length_2=filter_length_2,
                 filter_length_3=filter_length_3,
                 filter_length_4=filter_length_4,
                 filter_time_length=filter_time_length,
                 first_nonlin=first_nonlin,
                 first_pool_mode=first_pool_mode,
                 first_pool_nonlin=first_pool_nonlin,
                 later_nonlin=later_nonlin,
                 later_pool_mode=later_pool_mode,
                 later_pool_nonlin=later_pool_nonlin,
                 pool_time_length=pool_time_length,
                 pool_time_stride=pool_time_stride,
                 split_first_layer=split_first_layer,
                 stride_before_pool=True).create_network()
    elif model_name == 'shallow_smac':
        conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        final_conv_length = 22
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length,
                                conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                ).create_network()
    elif model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(n_chans, n_classes, (600,1)))
        model.add_module('softmax', nn.LogSoftmax(dim=1))
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    else:
        assert False, "unknown model name {:s}".format(model_name)

    to_dense_prediction_model(model)

    if config.cuda:
    	model.cuda()
    test_input = np_to_var(np.ones((2, config.n_chans, config.input_time_length, 1), dtype=np.float32))
    if config.cuda:
        test_input = test_input.cuda()

    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    iterator = CropsFromTrialsIterator(batch_size=config.batch_size,
                                           input_time_length=config.input_time_length,
                                           n_preds_per_input=n_preds_per_input)

    #model.add_module('softmax', nn.LogSoftmax(dim=1))

    model.eval()

    mode[2] = str(mode[2])
    mode[3] = str(mode[3])
    modelName = '-'.join(mode[:4])

    #params = th.load('sessionsData/%sModel%s-sessions.pt'%(modelName,mode[4]))
    #params = th.load('%sModel%s.pt'%(modelName,mode[4]))
    params = th.load('linear/%sModel%s.pt'%(modelName,mode[4]))


    model.load_state_dict(params)

    if config.test_on_eval:
        #test_X, test_y = test_dataset.load()
        #test_X, test_y = loadNEDCdata(mode='eval')
        max_shape = np.max([list(x.shape) for x in test_X],
                           axis=0)
        assert max_shape[1] == int(config.test_recording_mins *
                                   config.sampling_freq * 60)
    if not config.test_on_eval:
        splitter = TrainValidTestSplitter(config.n_folds, config.i_test_fold,
                                          shuffle=config.shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
    else:
        splitter = TrainValidSplitter(config.n_folds, i_valid_fold=config.i_test_fold,
                                          shuffle=config.shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        del test_X, test_y
    del X,y # shouldn't be necessary, but just to make sure

    datasets = OrderedDict((('train', train_set), ('valid', valid_set), ('test', test_set)))

    for setname in ('train', 'valid', 'test'):
        #setname = 'test'
        #print("Compute predictions for {:s}...".format(setname))
        dataset = datasets[setname]
        if config.cuda:
            preds_per_batch = [var_to_np(model(np_to_var(b[0]).cuda()))
                      for b in iterator.get_batches(dataset, shuffle=False)]
        else:
            preds_per_batch = [var_to_np(model(np_to_var(b[0])))
                      for b in iterator.get_batches(dataset, shuffle=False)]
        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=iterator.input_time_length,
            n_stride=iterator.n_preds_per_input)
        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                    preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        all_pred_labels = np.argmax(mean_preds_per_trial, axis=1).squeeze()
        all_target_labels = dataset.y
        acc_per_class = []
        for i_class in range(n_classes):
            mask = all_target_labels == i_class
            acc = np.mean(all_pred_labels[mask] ==
                          all_target_labels[mask])
            acc_per_class.append(acc)
        misclass = 1 - np.mean(acc_per_class)
        #print('Acc:{}, Class 0:{}, Class 1:{}'.format(np.mean(acc_per_class),acc_per_class[0],acc_per_class[1]))
        
        if setname == 'test':
            testResult = np.mean(acc_per_class)

    return testResult

        #pdb.set_trace()

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh - 0.05

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom', fontsize=24)
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

def autolabel(rects,labels,err,ax):
    i = 0
    for rect in rects:
        h = rect.get_height()
        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
        #        ha='center', va='bottom')
        s = labels[i]
        if i == 0:
            coeff = 1.0
        else:
            coeff = 0.99
        ax.text(rect.get_x()+rect.get_width()/2., coeff*(h+err[i]), '%s'%s,
                ha='center', va='bottom',fontsize=16)
        i += 1

def plotBarsDCNN():
    labels = ['Dilantin vs. Keppra', 'Dilantin vs. None', 'Keppra vs. None']
    accA = (60.39, 68.75, 68.85)
    errA = (5.03, 6.29, 6.87)
    accN = (59.12, 64.12, 64.71)
    errN = (7.49, 8.19, 6.31)
    ind = np.arange(3)
    width= 0.4

    fig,ax = plt.subplots()
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    barsN = plt.bar(ind, accN, **bar_kwargsN)
    errsN = plt.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    bar_kwargsA = {'width':width,'color':'g','linewidth':2,'zorder':5,'align':'center','fill':False,'hatch':'\\\\'}
    err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead

    barsA = plt.bar(ind+width, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind+width, accA, yerr=errA, **err_kwargsA)

    ax.legend( (barsN[0], barsA[0]), ('Normal EEG', 'Abnormal EEG'),loc='upper right',fontsize=14)

    # for f_oneway
    '''
    barplot_annotate_brackets(1, 2, 'n.s. - p = .976', ind+width, accA, dh=.1)
    barplot_annotate_brackets(0, 1, '** - p = .006', ind+width, accA, dh=.15)
    barplot_annotate_brackets(0, 2, '** - p = .008', ind+width, accA, dh=.195)
    
    autolabel(barsN,['*','***','***'],errN,ax)
    autolabel(barsA,['*','***','***'],errA,ax)
    '''


    plt.ylim(ymax=100)
    plt.xticks(ind+width/2, labels, color='k',fontsize=18)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classifications',fontsize=24)
    plt.title('DCNN results',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    plt.show()

def plotBarsabnorm():
    labels = ['Dilantin vs. Keppra', 'Dilantin vs. No medications', 'Keppra vs. No medications']
    accA = (60.39, 68.75, 68.85)
    errA = (5.03, 6.29, 6.87)
    accN = (59.23, 68.90, 70.00)
    errN = (6.25, 7.18, 4.14)
    ind = np.arange(3)
    width= 0.4

    fig,ax = plt.subplots()
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2,'barsabove':True}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    barsN = plt.bar(ind, accN, **bar_kwargsN)
    errsN = plt.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    bar_kwargsA = {'width':width,'color':'gray','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.5,'hatch':'\\\\'}
    err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2,'barsabove':True}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    
    #bp = ax.boxplot(accA)#,whis='range')#,whis='range'

    barsA = plt.bar(ind+width, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind+width, accA, yerr=errA, **err_kwargsA)

    ax.legend( (barsN[0], barsA[0]), ('kSVM', 'DCNN'),loc='upper right',fontsize=20)

    # for f_oneway

    barplot_annotate_brackets(1, 2, 'n.s. - p < .880', ind+width, accA, dh=.05)
    barplot_annotate_brackets(0, 1, '* - p < .006', ind+width, accA, dh=.06)
    barplot_annotate_brackets(0, 2, '* - p < .009', ind+width, accA, dh=.23)

    barplot_annotate_brackets(1, 2, 'n.s. - p < .649', ind, accN, dh=.11)
    barplot_annotate_brackets(0, 1, '* - p < .010', ind, accN, dh=.13)
    barplot_annotate_brackets(0, 2, '* - p < .002', ind, accN, dh=.24)

    barplot_annotate_brackets(0, 1, 'n.s.', (ind[0],ind[0]+width), (accN[0],accA[0]), dh=-.01)
    barplot_annotate_brackets(0, 1, 'n.s.', (ind[1],ind[1]+width), (accN[1],accA[1]), dh=-.01)
    barplot_annotate_brackets(0, 1, 'n.s.', (ind[2],ind[2]+width), (accN[0],accA[2]), dh=-.01)
    
    #autolabel(barsN,['*','***','***'],errN,ax)
    #autolabel(barsA,['n.s.','**','**'],errA,ax)


    plt.ylim(ymin=50,ymax=100)
    plt.xticks(ind+width/2, labels, color='k',fontsize=20)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classifications',fontsize=24)
    plt.title('Classification results of subjects with abnormal EEG',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    plt.show()


def plotBarsabnormAllMethods():
    labels = ['SVM', 'kSVM', 'RF', 'NN', 'SCNN', 'DCNN']
    '''
    accA = (57.50,66.72,71.54)
    errA = (3.69,6.78,71.54)
    accB = (57.88,70.78,73.46)
    errB = (5.99,3.28,3.92)
    accC = (64.23,70.00,70.00)
    errC = (6.58,2.95,4.80)
    accD = (50.77,52.66,48.84)
    errD = (6.09,5.05,5.17)
    accE = (56.35,64.53,70.77)
    errE = (6.02,5.68,3.42)
    accF = (60.19,68.59,68.65)
    errF = (4.94,6.11,6.66)
    '''
    
    accA = (57.50,57.88,64.23,50.77,56.35,60.19)
    errA = (3.69,5.99,6.58,6.09,6.02,4.94)
    accB = (66.72,70.78,70.00,52.66,64.53,68.59)
    errB = (6.78,3.28,2.95,3.05,5.68,6.11)
    accC = (71.54,73.46,70.00,48.84,70.77,68.65)
    errC = (5.76,3.92,4.80,5.17,3.42,6.66)

    ind = np.arange(6)
    width= 0.3

    fig,ax = plt.subplots()
    err_kwargs = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2,'barsabove':True}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    bar_kwargsA = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.25}
    barsA = plt.bar(ind, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind, accA, yerr=errA, **err_kwargs)

    bar_kwargsB = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.5}#,'hatch':'\\\\'}
    barsB = plt.bar(ind+width*1, accB, **bar_kwargsB)
    errsB = plt.errorbar(ind+width*1, accB, yerr=errB, **err_kwargs)

    bar_kwargsC = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.75}#,'hatch':'+'}
    barsC = plt.bar(ind+width*2, accC, **bar_kwargsC)
    errsC = plt.errorbar(ind+width*2, accC, yerr=errC, **err_kwargs)

    ax.legend((barsA[0], barsB[0], barsC[0]), ('Dilantin vs. Keppra', 'Dilantin vs. No medications', 'Keppra vs. No medications'),loc='upper right',fontsize=20)

    # for f_oneway


    barplot_annotate_brackets(0, 1, '*', (ind[0],ind[0]+width), (accA[0],accB[0]), dh=.05)
    barplot_annotate_brackets(0, 1, '**', (ind[0],ind[0]+width*2), (accA[0],accC[0]), dh=.04)

    barplot_annotate_brackets(0, 1, '**', (ind[1],ind[1]+width), (accA[1],accB[1]), dh=.01)
    barplot_annotate_brackets(0, 1, '**', (ind[1],ind[1]+width*2), (accA[1],accC[1]), dh=.01)

    barplot_annotate_brackets(0, 1, '*', (ind[4],ind[4]+width), (accA[4],accB[4]), dh=.04)
    barplot_annotate_brackets(0, 1, '**', (ind[4],ind[4]+width*2), (accA[4],accC[4]), dh=.01)

    barplot_annotate_brackets(0, 1, '*', (ind[5],ind[5]+width), (accA[5],accB[5]), dh=.03)
    barplot_annotate_brackets(0, 1, '*', (ind[5],ind[5]+width*2), (accA[5],accC[5]), dh=.06)

    #autolabel(barsN,['*','***','***'],errN,ax)
    #autolabel(barsA,['n.s.','**','**'],errA,ax)


    plt.ylim(ymin=50,ymax=100)
    plt.xticks(ind+width, labels, color='k',fontsize=20)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classification method',fontsize=24)
    plt.title('Comparison of results from subjects with abnormal EEG across classifications',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    #plt.show()

def plotBarsabnormAllClasses():
    labels = ['Dilantin vs. Keppra', 'Dilantin vs. No medications', 'Keppra vs. No medications']

    accA = (57.50,66.72,71.54)
    errA = (3.69,6.78,5.76)
    accB = (57.88,70.78,73.46)
    errB = (5.99,3.28,3.92)
    accC = (64.23,70.00,70.00)
    errC = (6.58,2.95,4.80)
    accD = (50.77,52.66,48.84)
    errD = (6.09,5.05,5.17)
    accE = (56.35,64.53,70.77)
    errE = (6.02,5.68,3.42)
    accF = (60.19,68.59,68.65)
    errF = (4.94,6.11,6.66)

    '''
    accA = (57.50,57.88,64.23,50.77,56.35,60.19)
    errA = (3.69,5.99,6.58,6.09,6.02,4.94)
    accB = (66.72,70.78,70.00,52.66,64.53,68.59)
    errB = (6.78,3.28,2.95,3.05,5.68,6.11)
    accC = (71.54,73.46,70.00,48.84,70.77,68.65)
    errC = (5.76,3.92,4.80,5.17,3.42,6.66)
    '''
    ind = np.arange(3)
    width= 0.18

    fig,ax = plt.subplots()
    err_kwargs = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2,'barsabove':True}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    bar_kwargsA = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.2}
    barsA = plt.bar(ind, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind, accA, yerr=errA, **err_kwargs)

    bar_kwargsB = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.35}#,'hatch':'\\\\'}
    barsB = plt.bar(ind+width*1, accB, **bar_kwargsB)
    errsB = plt.errorbar(ind+width*1, accB, yerr=errB, **err_kwargs)

    bar_kwargsC = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.5}#,'hatch':'+'}
    barsC = plt.bar(ind+width*2, accC, **bar_kwargsC)
    errsC = plt.errorbar(ind+width*2, accC, yerr=errC, **err_kwargs)

    bar_kwargsE = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.65}#,'hatch':'x'}
    barsE = plt.bar(ind+width*3, accE, **bar_kwargsE)
    errsE = plt.errorbar(ind+width*3, accE, yerr=errE, **err_kwargs)

    bar_kwargsF = {'width':width,'color':'k','linewidth':2,'zorder':5,'align':'center','fill':True,'alpha':0.8}#,'hatch':'.'}
    barsF = plt.bar(ind[1:]+width*4, accF[1:], **bar_kwargsF)
    errsF = plt.errorbar(ind[1:]+width*4, accF[1:], yerr=errF[1:], **err_kwargs)

    ax.legend((barsA[0], barsB[0], barsC[0], barsE[0], barsF[1]), ('SVM', 'kSVM', 'RF', 'SCNN', 'DCNN'),loc='upper right',fontsize=20)

    # for f_oneway


    barplot_annotate_brackets(0, 1, '*', (ind[0],ind[0]+width*2), (accA[0],accC[0]), dh=.09)
    #barplot_annotate_brackets(0, 1, '*', (ind[0]+width,ind[0]+width*2), (accB[0],accC[0]), dh=.05)
    #barplot_annotate_brackets(0, 1, '*', (ind[0]+width*3,ind[0]+width*2), (accE[0],accC[0]), dh=.07)

    #barplot_annotate_brackets(0, 1, '*', (ind[1]+width*3,ind[1]+width), (accE[1],accB[1]), dh=.04)
    #barplot_annotate_brackets(0, 1, '*', (ind[1]+width*3,ind[1]+width*2), (accE[1],accC[1]), dh=.01)


    #autolabel(barsN,['*','***','***'],errN,ax)
    #autolabel(barsA,['n.s.','**','**'],errA,ax)


    plt.ylim(ymin=50,ymax=100)
    plt.xticks((ind[0]+1.5*width,ind[1]+2*width,ind[2]+2*width), labels, color='k',fontsize=20)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classification',fontsize=24)
    plt.title('Comparison of results from subjects with abnormal EEG across methods',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.08,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    #plt.show()

def plotBarsDCNNabnorm():
    labels = ['Dilantin vs. Keppra', 'Dilantin vs. No medications', 'Keppra vs. No medications']
    accA = (60.39, 68.75, 68.85)
    errA = (5.03, 6.29, 6.87)
    accN = (59.12, 64.12, 64.71)
    errN = (7.49, 8.19, 6.31)
    ind = np.arange(3)
    width= 0.8

    fig,ax = plt.subplots()
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    #bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    #err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #barsN = plt.bar(ind, accN, **bar_kwargsN)
    #errsN = plt.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    bar_kwargsA = {'width':width,'color':'g','linewidth':2,'zorder':5,'align':'center','fill':False}#,'hatch':'\\\\'}
    err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    
    #bp = ax.boxplot(accA)#,whis='range')#,whis='range'

    barsA = plt.bar(ind, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind, accA, yerr=errA, **err_kwargsA)

    #ax.legend( (barsN[0], barsA[0]), ('Normal EEG', 'Abnormal EEG'),loc='upper right',fontsize=14)

    # for f_oneway

    barplot_annotate_brackets(1, 2, 'n.s. - p = .880', ind, accA, dh=.1)
    barplot_annotate_brackets(0, 1, '* - p = .005', ind, accA, dh=.15)
    barplot_annotate_brackets(0, 2, '* - p = .009', ind, accA, dh=.195)
    
    #autolabel(barsN,['*','***','***'],errN,ax)
    #autolabel(barsA,['n.s.','**','**'],errA,ax)


    plt.ylim(ymax=100)
    plt.xticks(ind, labels, color='k',fontsize=20)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classifications',fontsize=24)
    plt.title('DCNN results of subjects with abnormal EEG',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    plt.show()

def plotBarsDCNNabnormSessions():
    labels = ['Dilantin vs. Keppra', 'Dilantin vs. No medications', 'Keppra vs. No medications']
    accA = (53.86, 60.19, 60.00)
    errA = (6.75, 5.83, 5.20)
    accN = (56.92, 66.33, 56.92)
    errN = (8.74, 5.67, 11.25)
    ind = np.arange(3)
    width= 0.8

    fig,ax = plt.subplots()
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    #bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    #err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #barsN = plt.bar(ind, accN, **bar_kwargsN)
    #errsN = plt.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    bar_kwargsA = {'width':width,'color':'g','linewidth':2,'zorder':5,'align':'center','fill':False}#,'hatch':'\\\\'}
    err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    
    #bp = ax.boxplot(accA)#,whis='range')#,whis='range'

    barsA = plt.bar(ind, accA, **bar_kwargsA)
    errsA = plt.errorbar(ind, accA, yerr=errA, **err_kwargsA)

    #ax.legend( (barsN[0], barsA[0]), ('Normal EEG', 'Abnormal EEG'),loc='upper right',fontsize=14)

    # for f_oneway

    barplot_annotate_brackets(1, 2, 'n.s. - p = .880', ind, accA, dh=.1)
    barplot_annotate_brackets(0, 1, 'n.s. - p = .049', ind, accA, dh=.15)
    barplot_annotate_brackets(0, 2, 'n.s. - p = .057', ind, accA, dh=.195)
    
    #autolabel(barsN,['*','***','***'],errN,ax)
    #autolabel(barsA,['n.s.','**','**'],errA,ax)


    plt.ylim(ymax=100)
    plt.xticks(ind, labels, color='k',fontsize=20)#,rotation=45)
    plt.yticks(fontsize=18)
    plt.ylabel('Mean test percent accuracy',fontsize=24)
    plt.xlabel('Classifications',fontsize=24)
    plt.title('DCNN results of subjects with abnormal EEG',fontsize=28)

    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.97,top=0.92,wspace=0.2,hspace=0.2)


    plt.show()

if __name__ == '__main__':

    startGlobal = time.time()

    classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','keppra',0],['none','dilantin',1],['none','keppra',1]]

    classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]

    #classes = [['dilantin','keppra',1],['none','dilantin',1],['none','keppra',1]]

    runModels = 0
    plotRes = 1

    if runModels == 1:

        allResults = []

        for mode in classes:
            modeResult = []
            modeResultRand = []
            for r in range(1,11):
                curMode = [mode[0],mode[1],mode[2],r,'']
                #print(curMode)
                curResult = runModel(curMode)
                #continue
                modeResult.append(curResult)

                curModeRand = [mode[0],mode[1],mode[2],r,'-shuffle']
                #print(curModeRand)
                curResultRand = runModel(curModeRand)
                modeResultRand.append(curResultRand)
            #continue
            allResults.append(modeResult)

            print('%s - Acc: %f (%f)'%(curMode,np.mean(modeResult),np.std(modeResult)))
            print('%s - Acc: %f (%f)'%(curModeRand,np.mean(modeResultRand),np.std(modeResultRand)))
            
            F,p = stats.kruskal(modeResult,modeResultRand) 
            print('%s - p =%f'%(curMode,p))

        toCompare = [[0,2],[0,3],[1,4],[1,5],[2,3],[4,5]]

        toCompare = [[0,2],[0,4],[1,3],[1,5],[2,4],[3,5]]

        #toCompare = [[0,1],[0,2],[1,2]]

        for c in toCompare:
            F,p = stats.kruskal(allResults[c[0]],allResults[c[1]]) 

            print('Significance: %s v. %s - p =%f'%(classes[c[0]],classes[c[1]],p))

        np.save('linearNN-recs-results',allResults)

    if plotRes == 1:

        svm = np.load('linear-recs-results.npy',allow_pickle=True)
        ksvm = np.load('rbf-recs-results.npy',allow_pickle=True)
        rf = np.load('rf-recs-results.npy',allow_pickle=True)

        nn = np.load('linearNN-recs-results.npy',allow_pickle=True)
        scnn = np.load('SCNN-recs-results.npy',allow_pickle=True)
        dcnn = np.load('DCNN-recs-results.npy',allow_pickle=True)


        for modeNum in range(len(classes)):
            F,p = stats.kruskal(svm[modeNum]/100.,ksvm[modeNum]/100.) 
            print('Significance: %s svm v. ksvm - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(svm[modeNum]/100.,rf[modeNum]/100.) 
            print('Significance: %s svm v. rf - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(svm[modeNum]/100.,nn[modeNum]) 
            print('Significance: %s svm v. nn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(svm[modeNum]/100.,scnn[modeNum]) 
            print('Significance: %s svm v. scnn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(svm[modeNum]/100.,dcnn[modeNum]) 
            print('Significance: %s svm v. dcnn - p =%f'%(classes[modeNum],p))
            
            F,p = stats.kruskal(ksvm[modeNum]/100.,rf[modeNum]/100.) 
            print('Significance: %s ksvm v. rf - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(ksvm[modeNum]/100.,nn[modeNum]) 
            print('Significance: %s ksvm v. nn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(ksvm[modeNum]/100.,scnn[modeNum]) 
            print('Significance: %s ksvm v. scnn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(ksvm[modeNum]/100.,dcnn[modeNum]) 
            print('Significance: %s ksvm v. dcnn - p =%f'%(classes[modeNum],p))

            F,p = stats.kruskal(rf[modeNum]/100.,nn[modeNum]) 
            print('Significance: %s rf v. nn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(rf[modeNum]/100.,scnn[modeNum]) 
            print('Significance: %s rf v. scnn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(rf[modeNum]/100.,dcnn[modeNum]) 
            print('Significance: %s rf v. dcnn - p =%f'%(classes[modeNum],p))
            
            F,p = stats.kruskal(nn[modeNum],scnn[modeNum]) 
            print('Significance: %s nn v. scnn - p =%f'%(classes[modeNum],p))
            F,p = stats.kruskal(nn[modeNum],dcnn[modeNum]) 
            print('Significance: %s nn v. dcnn - p =%f'%(classes[modeNum],p))

            F,p = stats.kruskal(scnn[modeNum],dcnn[modeNum]) 
            print('Significance: %s scnn v. dcnn - p =%f'%(classes[modeNum],p))

        svm = np.array(svm)/100.
        svmMean = np.mean(np.mean(svm,axis=1),axis=0)
        svmSD = np.mean(np.std(svm,axis=1),axis=0)

        ksvm = np.array(ksvm)/100.
        ksvmMean = np.mean(np.mean(ksvm,axis=1),axis=0)
        ksvmSD = np.mean(np.std(ksvm,axis=1),axis=0)

        rf = np.array(rf)/100.
        rfMean = np.mean(np.mean(rf,axis=1),axis=0)
        rfSD = np.mean(np.std(rf,axis=1),axis=0)

        nn = np.array(nn)
        nnMean = np.mean(np.mean(nn,axis=1),axis=0)
        nnSD = np.mean(np.std(nn,axis=1),axis=0)

        scnn = np.array(scnn)
        scnnMean = np.mean(np.mean(scnn,axis=1),axis=0)
        scnnSD = np.mean(np.std(scnn,axis=1),axis=0)

        dcnn = np.array(dcnn)
        dcnnMean = np.mean(np.mean(dcnn,axis=1),axis=0)
        dcnnSD = np.mean(np.std(dcnn,axis=1),axis=0)
        
        print('SVM Acc Mean (SD): %0.4f (%0.4f)'%(svmMean,svmSD))
        print('kSVM Acc Mean (SD): %0.4f (%0.4f)'%(ksvmMean,ksvmSD))
        print('RF Acc Mean (SD): %0.4f (%0.4f)'%(rfMean,rfSD))
        print('NN Acc Mean (SD): %0.4f (%0.4f)'%(nnMean,nnSD))
        print('SCNN Acc Mean (SD): %0.4f (%0.4f)'%(scnnMean,scnnSD))
        print('DCNN Acc Mean (SD): %0.4f (%0.4f)'%(dcnnMean,dcnnSD))
        
        featuresMean = [svmMean,ksvmMean,rfMean]
        featuresSD = [svmSD,ksvmSD,rfSD]

        deepMean = [nnMean,scnnMean,dcnnMean]
        deepSD = [nnSD,scnnSD,dcnnSD]

        print('Mean Difference: , %0.4f SD Difference:%0.4f (feature-deep)'%(np.mean(featuresMean)-np.mean(deepMean),np.mean(featuresSD)-np.mean(deepSD)))

        #plotBarsDCNN()
        
        #plotBarsabnorm()
        
        plotBarsabnormAllMethods()
        plotBarsabnormAllClasses()
        plt.show()

        #plotBarsDCNNabnorm()
        #plotBarsDCNNabnormSessions()
        #pdb.set_trace()

    endGlobal = time.time()

    print('Time elapsed for all tests: %s'%(str(endGlobal-startGlobal)))