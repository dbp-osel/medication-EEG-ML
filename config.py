#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration file for EEG-driven deep learning classifications and model generation
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

# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = [
    '/media/david/Data1/normal-abnormal/normal/',
    '/media/david/Data1/normal-abnormal/abnormal/']
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
model_name = 'deep'#'deep'#'shallow' 'linear'
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 6000
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = True # False
