## Application of FCNs & GCNs to fmri movie data (camcan study)  
#Inspect accuracy results/confusion matrices across time

#Imports
print('Start')
task_type =  'Rest' #'Rest'
reverse_direction = False 
import os
import sys
import math
import time
import datetime

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

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

#Relaoding of 
#from models import model_fit_evaluate as model_fit_evaluate 
#import importlib
#importlib.reload(models.model_fit_evaluate)
#importlib.reload(models.test)

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
fmri, subj_list = get_fmri_data(root_pth, task_type)
print(np.array(fmri).shape) #(193, 400)
print(len(subj_list)) #644 subjects
#fmri_copy = fmri.copy()

#Match fmri movie shape
fmri = fmri[:,:193,:]
fmri.shape

#Reverse direction
if reverse_direction:
    print('DIRECTION REVERSED')
    fmri = np.flip(fmri, axis = 1)
    print(f'fmri shape = {fmri.shape}')

#Adjacency matrix
#root_pth = '/camcan/schaefer_parc/'
#adj_mat = get_rsfmri_adj_matrix(root_pth) #Should save this rather then recalculating 
#adj_mat = Adjacency_matrix(adj_mat, n_neighbours = 8).get_adj_sp_torch_tensor()

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

#************************************************************
#2. Dataloader
params = {'batch_size': 1,  #2
          'shuffle': True,
          'num_workers': 2}
          
#Split into train and test 
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

#*************************************
#3. Compare train + test trends
#i. Loop -> matrix of train + test
#ii. Average value for each class (32)
#ii. Plot train vs test 

#Train
fmri_train_matrix = []
for i in np.arange(0,len(train_idx)): #Loop through subjects
    fmri_train_matrix.append(fmri_train.__getitem__(i)[0].numpy())

fmri_train_matrix = np.array(fmri_train_matrix)

#Average over subjects, time within a block, ROIs 
fmri_train_avg_across_blocks = np.mean(np.mean(np.mean(fmri_train_matrix, axis = 0), axis = 1), axis = 1)

#Test
fmri_test_matrix = []
for i in np.arange(0,len(test_idx)): #Loop through subjects
    fmri_test_matrix.append(fmri_test.__getitem__(i)[0].numpy())

fmri_test_matrix = np.array(fmri_test_matrix)

#Average over subjects, time within a block, ROIs
fmri_test_avg_across_blocks = np.mean(np.mean(np.mean(fmri_test_matrix, axis = 0), axis = 1), axis = 1)


#***************************************************************
#Model - Inspect Accuracy results
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

best_acc, best_confusion_matrix, best_predictions, best_target_classes, best_prop, best_count  = model_fit_evaluate(model, adj_mat, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs)

#Save results to file
#version = 1
#with open('best_confusion_matrix{}.txt'.format(version), 'w') as f:
#    for row in best_confusion_matrix:
#        f.write(' '.join([str(a) for a in row]) + '\n')

#**************************
#Graphical model
from model import ChebNet

print(f'Block duration = {block_duration}')
#Model params 
loss_func = nn.CrossEntropyLoss()
filters = 32; num_layers = 2
num_epochs = 10 
model_test = ChebNet(block_duration, filters, n_labels, gcn_layer = num_layers,dropout=0.25,gcn_flag=True)
#model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)

model_test = model_test.to(device)
adj_mat = adj_mat.to(device)
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))

optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)

gcn_best_acc, gcn_best_confusion_matrix, gcn_best_predictions, gcn_best_target_classes, gcn_best_prop, gcn_best_count = model_fit_evaluate(model_test, adj_mat, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs)

#*******************
#Fifth order GCN

#Model params 
k_order = 5
print(f'Block duration = {block_duration}')
loss_func = nn.CrossEntropyLoss()
filters = 32; num_layers = 2
num_epochs = 10 

#Model
model_test = ChebNet(block_duration, filters, n_labels, K=k_order, gcn_layer=num_layers,dropout=0.25)

model_test = model_test.to(device)
#adj_mat = adj_mat.to(device)
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))

optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)

gcn_best_acc5, gcn_best_confusion_matrix5, gcn_best_predictions5, gcn_best_target_classes5, gcn_best_prop5, gcn_best_count5 = model_fit_evaluate(model_test, adj_mat, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs)
