## Application of FCNs & GCNs to fmri movie data (camcan study)  
#Inspect accuracy results/confusion matrices across time

#Imports
print('Start')
import os
import sys
import math
import time
import datetime

import numpy as np
import pandas as pd
import scipy.io

from scipy import sparse
from scipy.stats import spearmanr
from sklearn import preprocessing, metrics,manifold
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
import itertools


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from util_funcs import *
from models import *
#reload(model)

import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings(action='once')
#%load_ext autoreload #%autoreload #%reload_ext autoreload #import sys #reload(sys)

#imports
#pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html

#CPU
device = torch.device("cpu")
print(device)

#****************************************
#Data
root_pth = '/camcan/schaefer_parc/'
task_type = 'Rest'
fmri, subj_list = get_fmri_data(root_pth, task_type)
print(np.array(fmri).shape) #(193, 400)
print(len(subj_list)) #644 subjects
#fmri_copy = fmri.copy()

#Match fmri movie shape
fmri = fmri[:,:193,:]
fmri.shape

#Adjacency matrix
root_pth = '/camcan/schaefer_parc/'
adj_mat = get_rsfmri_adj_matrix(root_pth) #Should save this rather then recalculating 
adj_mat = Adjacency_matrix(adj_mat, n_neighbours = 8).get_adj_sp_torch_tensor()

#Model/fmri paratmeters
TR = 2.47
n_subjects = np.array(fmri).shape[0]
print(f'N subjects = {n_subjects}')
n_regions = np.array(fmri).shape[2] 
print(n_regions)

#Specify block duration
block_duration = 6 #8 #16 #6 8 -Factor of 192 
total_time = fmri.shape[1]
n_blocks = total_time // block_duration
n_labels = n_blocks
print(f'Number of blocks = {n_blocks}')
total_time = block_duration*n_blocks #Rounded number 
print(f'Total time = {total_time}')

#*************
#Data preprocessing - filter + normalise fmri
def filter_fmri(fmri, standardize):  
    'filter fmri signal'
    
    #fmri
    fmri_filtered = []
    for subj in np.arange(0, fmri.shape[0]):
        fmri_subj = fmri[subj]
        filtered = nilearn.signal.clean(fmri_subj, sessions= None, detrend=True, 
                               standardize= False, confounds=None, low_pass= 0.1, 
                               high_pass= 0.01, t_r = TR, ensure_finite=False)
        fmri_filtered.append(filtered)
    
    return fmri_filtered

#Apply
standardize =  'zscore' #'psc', False
fmri_filtered = filter_fmri(fmri, standardize)
print(np.array(fmri_filtered).shape)

#***************
#Dataloader
#Split into train and test 
params = {'batch_size': 2, 
          'shuffle': True,
          'num_workers': 2}
test_size = 0.2
randomseed= 12345
rs = np.random.RandomState(randomseed)

#Training/Test indices
train_idx, test_idx = train_test_split(range(n_subjects), test_size = test_size, random_state=rs, shuffle=True)
print('Training on %d subjects, Testing on %d subjects' % (len(train_idx), len(test_idx)))

#Train set
print(f'Block duration = {block_duration}')
fmri_data_train = [fmri_filtered[i] for i in train_idx] #Training subjects 
print(np.array(fmri_data_train).shape)
fmri_train = Fmri_dataset(fmri_data_train, TR, block_duration)
train_loader = DataLoader(fmri_train, collate_fn = fmri_samples_collate_fn, **params)

#Test set
fmri_data_test = [fmri_filtered[i] for i in test_idx]
print(np.array(fmri_data_test).shape)
fmri_test = Fmri_dataset(fmri_data_test, TR, block_duration)
test_loader = DataLoader(fmri_test, collate_fn=fmri_samples_collate_fn, **params)

#***************************************************************
#Inspect Accuracy results

#Block duration
print(f'Block duration = {block_duration}')

#Define model 
model = FCN(n_regions, n_labels) #time points == x, regions == rows 
model = model.to(device)
print(model)
print("{} paramters to be trained in the model\n".format(count_parameters(model)))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()
num_epochs=10
#adj_mat = 'a'

best_acc, best_confusion_matrix = model_fit_evaluate(model, adj_mat, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs)

#Save results to file
with open('best_confusion_matrix{}.txt'.format(version), 'w') as f:
    for row in best_confusion_matrix:
        f.write(' '.join([str(a) for a in row]) + '\n')