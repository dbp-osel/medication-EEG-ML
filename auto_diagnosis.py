#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EEG-driven deep learning classifications and model generation
Software tools used generate results in submitted work to (tentative citation pending review):

David O. Nahmias and Kimberly L. Kontson. 
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
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import pdb
from loadNEDC import loadNEDCdata,loadSubNormData,addDataNoise

if len(sys.argv)>1:
    if ',' in sys.argv[1]:
        CLASSY = sys.argv[1].split(',')
        CLASSY[2] = int(CLASSY[2])
        if len(CLASSY) == 4:
            CLASSY[3] = int(CLASSY[3])
    else:
        CLASSY = str(sys.argv[1])

def splitDataRandom(allData,allLabels,setNum=0,shuffle=0):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = allData[allLabels == 0]
    data1 = allData[allLabels == 1]


    fullRange = list(range(numberEqSamples))
    random.shuffle(fullRange)


    testIndecies = fullRange[trainSamplesNum:]
    trainIndecies = fullRange[:trainSamplesNum]

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)
    '''
    if shuffle == 1:
        random.shuffle(allLabelsTrain)
        #np.save('sessionsData/trainLabels%s-shuffle-sessions'%CLASSY,allLabelsTrain)
        #np.save('sessionsData/dataRangeOrder%s-shuffle-sessions'%CLASSY,fullRange)
        
        np.save('deep/trainLabels%s-shuffle'%CLASSY,allLabelsTrain)
        np.save('deep/dataRangeOrder%s-shuffle'%CLASSY,fullRange)
    else:
        #np.save('sessionsData/dataRangeOrder%s-sessions'%CLASSY,fullRange)
        
        np.save('deep/dataRangeOrder%s'%CLASSY,fullRange)
    '''
    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest


def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set


def run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs,cuda,):
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=data_folders,
                           train_or_eval='train',
                           sensor_types=sensor_types)
    if test_on_eval:
        if test_recording_mins is None:
            test_recording_mins = duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                max_recording_mins=None,
                                preproc_functions=test_preproc_functions,
                                data_folders=data_folders,
                                train_or_eval='eval',
                                sensor_types=sensor_types)
    #X,y = dataset.load()
    #test_X, test_y = test_dataset.load()

    #X,y = loadNEDCdata(mode='train')
    print(CLASSY)
    #X,y,test_X,test_y = loadNEDCdata(mode='all',classy=CLASSY)

    #data = np.load('sessionsData/data%s-sessions.npy'%CLASSY[:3])
    #labels = np.load('sessionsData/labels%s-sessions.npy'%CLASSY[:3])

    data = np.load('data%s.npy'%CLASSY[:3])
    labels = np.load('labels%s.npy'%CLASSY[:3])

    #pdb.set_trace()

    if len(CLASSY) > 3:
        random.seed(11081992 + int(CLASSY[3]))
    else:
        random.seed(11081992)

    X,y,test_X,test_y = splitDataRandom(data,labels,shuffle=0)

    #X = np.load('trainData%s.npy'%CLASSY)
    #y = np.load('trainLabels%s.npy'%CLASSY)
    #test_X = np.load('testData%s.npy'%CLASSY)
    #test_y = np.load('testLabels%s.npy'%CLASSY)

    #return False

    #pdb.set_trace()

    #X,y,test_X,test_y = loadSubNormData(mode='all')

    #X = addDataNoise(X,band=[1,4])
    #test_X = addDataNoise(test_X,band=[1,4])
    #pdb.set_trace()

    
    max_shape = np.max([list(x.shape) for x in X],
                       axis=0)
    assert max_shape[1] == int(duration_recording_mins *
                               sampling_freq * 60)
    if test_on_eval:
        #test_X, test_y = test_dataset.load()
        #test_X, test_y = loadNEDCdata(mode='eval')
        max_shape = np.max([list(x.shape) for x in test_X],
                           axis=0)
        assert max_shape[1] == int(test_recording_mins *
                                   sampling_freq * 60)
    if not test_on_eval:
        splitter = TrainValidTestSplitter(n_folds, i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
    else:
        splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        del test_X, test_y
    del X,y # shouldn't be necessary, but just to make sure

    #pdb.set_trace()

    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
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
    log.info("Model:\n{:s}".format(str(model)))
    if cuda:
        model.cuda()
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    log.info("In shape: {:s}".format(str(test_input.cpu().data.numpy().shape)))

    out = model(test_input)
    log.info("Out shape: {:s}".format(str(out.cpu().data.numpy().shape)))
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedDiagnosisMonitor(input_time_length, n_preds_per_input),
                RuntimeMonitor(),]
    stop_criterion = MaxEpochs(max_epochs)
    batch_modifier = None
    run_after_early_stop = True
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop,
                     batch_modifier=batch_modifier,
                     cuda=cuda)
    exp.run()
    return exp



if __name__ == "__main__":
    import config
    print('Classifying: {}'.format(CLASSY))
    #pdb.set_trace()
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    exp = run_exp(
        config.data_folders,
        config.n_recordings,
        config.sensor_types,
        config.n_chans,
        config.max_recording_mins,
        config.sec_to_cut, config.duration_recording_mins,
        config.test_recording_mins,
        config.max_abs_val,
        config.sampling_freq,
        config.divisor,
        config.test_on_eval,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.n_start_chans, config.n_chan_factor,
        config.input_time_length, config.final_conv_length,
        config.model_constraint,
        config.init_lr,
        config.batch_size, config.max_epochs,config.cuda,)
    end_time = time.time()
    run_time = end_time - start_time

    log.info("Experiment runtime: {:.2f} sec".format(run_time))
    
    #pdb.set_trace()
    if len(CLASSY) == 3:
        CLASSY[2] = str(CLASSY[2])
        CLASSY = '-'.join(CLASSY)
    elif len(CLASSY) == 4:       
        CLASSY[2] = str(CLASSY[2])
        CLASSY[3] = str(CLASSY[3])
        CLASSY = '-'.join(CLASSY)

    #th.save(exp.model.state_dict(),'sessionsData/'+CLASSY+'Model-sessions.pt')
    
    #th.save(exp.model.state_dict(),'deep/'+CLASSY+'Model.pt')
    #th.save(exp.model.state_dict(),'deep/'+CLASSY+'Model-shuffle.pt')

    #savedModel = th.load('subNormModel.pt')
    
    # In case you want to recompute predictions for further analysis:
    exp.model.eval()
    for setname in ('train', 'valid', 'test'):
        log.info("Compute predictions for {:s}...".format(
            setname))
        dataset = exp.datasets[setname]
        if config.cuda:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                      for b in exp.iterator.get_batches(dataset, shuffle=False)]
        else:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                      for b in exp.iterator.get_batches(dataset, shuffle=False)]
        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=exp.iterator.input_time_length,
            n_stride=exp.iterator.n_preds_per_input)
        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                    preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

