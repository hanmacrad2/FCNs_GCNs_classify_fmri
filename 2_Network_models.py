# Application of FCNs & GCNs to fmri movie data (camcan study)
# Inspect accuracy results/confusion matrices across time

# Imports
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn import preprocessing, metrics, manifold
import warnings
import matplotlib.pyplot as plt
import importlib
from models import *
from util_funcs import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
from scipy.stats import spearmanr
from scipy import sparse
import scipy.io
import pandas as pd
import numpy as np
import datetime
import time
import math
import sys
import os
print('Start')


#importlib.reload(models.model_fit_evaluate)
#importlib.reload(models.test)


# %matplotlib inline
warnings.filterwarnings(action='once')
# %load_ext autoreload #%autoreload #%reload_ext autoreload #import sys #reload(sys)

# imports
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html

# CPU
device = torch.device("cpu")
print(device)

# ****************************************
# Data
root_pth = '/camcan/schaefer_parc/'
task_type = 'Movie'
fmri, subj_list = get_fmri_data(root_pth, task_type)
print(np.array(fmri).shape)  # (193, 400)
print(len(subj_list))  # 644 subjects
#fmri_copy = fmri.copy()

# Match fmri movie shape
fmri = fmri[:, :193, :]
fmri.shape

# Adjacency matrix
root_pth = '/camcan/schaefer_parc/'
# Should save this rather then recalculating
adj_mat = get_rsfmri_adj_matrix(root_pth)
adj_mat = Adjacency_matrix(adj_mat, n_neighbours=8).get_adj_sp_torch_tensor()

# Model/fmri paratmeters
TR = 2.47
n_subjects = np.array(fmri).shape[0]
print(f'N subjects = {n_subjects}')
n_regions = np.array(fmri).shape[2]
print(n_regions)

# Specify block duration
block_duration = 6  # 8 #16 #6 8 -Factor of 192
total_time = fmri.shape[1]
n_blocks = total_time // block_duration
n_labels = n_blocks
print(f'Number of blocks = {n_blocks}')
total_time = block_duration*n_blocks  # Rounded number
print(f'Total time = {total_time}')

# *************
# Data preprocessing - filter + normalise fmri


def filter_fmri(fmri, standardize):
    'filter fmri signal'

    # fmri
    fmri_filtered = []
    for subj in np.arange(0, fmri.shape[0]):
        fmri_subj = fmri[subj]
        filtered = nilearn.signal.clean(fmri_subj, sessions=None, detrend=True,
                                        standardize=False, confounds=None, low_pass=0.1,
                                        high_pass=0.01, t_r=TR, ensure_finite=False)
        fmri_filtered.append(filtered)

    return np.array(fmri_filtered)


# Apply
standardize = 'zscore'  # 'psc', False
fmri_filtered = filter_fmri(fmri, standardize)
print(np.array(fmri_filtered).shape)
#fmri_filtered = np.array(fmri_filtered)

# *****************************************
# Split into networks
class Network_Model():
    'Fully Connected Network'

    def __init__(self, fmri, adj_mat, network_file, block_duration):
        super(Network_Model, self).__init__()

        # Data
        self.fmri = fmri
        self.adj_mat = adj_mat
        self.network_file = network_file
        self.list_networks = ['Vis', 'SomMot', 'DorsAttn','VentAttn', 'Limbic', 'Cont', 'Default']
        #self.df_network = self.create_network_data(self.network_file, self.list_networks)

        # fmri params
        self.TR = 2.47
        self.n_regions = np.array(fmri).shape[2]
        self.n_subjects = fmri.shape[0]
        self.block_duration = block_duration  # 8 #16 #6 8 -Factor of 192
        self.total_time = fmri.shape[1]
        self.n_blocks = total_time // block_duration
        self.n_labels = n_blocks

        # Model params
        self.device = torch.device("cpu")
        self.params = {'batch_size': 1,
                       'shuffle': True,
                       'num_workers': 2}
        self.test_size = 0.2
        self.randomseed = 12345
        self.rs = np.random.RandomState(self.randomseed)
        #self.df_results = pd.DataFrame()

    def create_network_data(self): #, list_networks):
        
        self.df_network = pd.read_csv(self.network_file, delimiter="\t")
        self.df_network.columns = ['index', 'network_name', 'x3', 'x4', 'x5', 'x6']
        self.df_network['network'] = self.df_network['network_name']
        # Subset networks
        for netw in self.list_networks:
            self.df_network.loc[self.df_network['network'].str.contains(netw), 'network'] = netw

        self.df_network

    def get_network_fmri(self, networkX):

        indx_netw = self.df_network.index[df["network"] == networkX].tolist()
        fmri_network = self.fmri[:, :, indx_netw]
        print(f'{networkX} network fmri shape = {fmri_network.shape}')

        # Num regioins
        n_regions = fmri_network.shape[2]
        return fmri_network, n_regions

    def get_train_test_data(self, fmri_network):
        
        # Training/Test indices
        train_idx, test_idx = train_test_split(range(self.n_subjects), test_size= self.test_size, random_state= self.rs, shuffle=True)
        print('Training on %d subjects, Testing on %d subjects' %
              (len(train_idx), len(test_idx)))

        # Train set
        fmri_data_train = [fmri_network[i] for i in train_idx]  # Training subjects
        fmri_train = Fmri_dataset(fmri_data_train, self.TR, self.block_duration)
        train_loader = DataLoader(fmri_train, collate_fn=fmri_samples_collate_fn, **self.params)

        # Test set
        fmri_data_test = [fmri_network[i] for i in test_idx]
        fmri_test = Fmri_dataset(fmri_data_test, self.TR, self.block_duration)
        test_loader = DataLoader(
            fmri_test, collate_fn=fmri_samples_collate_fn, **self.params)

        return train_loader, test_loader

    def get_df_results_networks(self):

        #Setup
        list_acc = []; list_prop = []
        df_network = self.create_network_data()
        df_results = pd.DataFrame()

        for networkX in self.list_networks:
            print(f'Network = {networkX}')
            fmri_networkX, n_regions = self.get_network_fmri(networkX)  # fmri or self.fmri??
            best_acc, best_prop = self.run_model(
                fmri_networkX, n_regions)
            list_acc.append(best_acc)
            list_prop.append(best_prop)

        # Dataframe
        df_results['network'] = self.list_networks
        df_results['accuracy'] = list_acc
        df_results['proportion'] = list_prop

        return df_results

    def run_model(self, fmri_networkX, n_regions):

        #Data
        train_loader, test_loader = self.get_train_test_data(fmri_networkX)

        #Model 
        model = FCN(n_regions, self.n_labels)  # time points == x, regions == rows
        model = model.to(self.device)
        print(model)
        print("{} paramters to be trained in the model\n".format(
            count_parameters(model)))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_func = nn.CrossEntropyLoss()
        num_epochs = 10

        best_acc, best_confusion_matrix, best_predictions, best_target_classes, best_prop, best_count = model_fit_evaluate(
            model, self.adj_mat, self.device, train_loader, test_loader, self.n_labels, optimizer, loss_func, num_epochs)

        return best_acc, best_prop



# Apply
network_file = 'networks_7_parcel_400.txt'
netw_model = Network_Model(fmri_filtered, adj_mat, network_file, block_duration)

df_results = netw_model.get_df_results_networks()









# ***************
# Dataloader
# Split into train and test
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 2}
test_size = 0.2
randomseed = 12345
rs = np.random.RandomState(randomseed)


# Training/Test indices
train_idx, test_idx = train_test_split(range(n_subjects), test_size=test_size, random_state=rs, shuffle=True)
print('Training on %d subjects, Testing on %d subjects' %
      (len(train_idx), len(test_idx)))

# Train set
print(f'Block duration = {block_duration}')
fmri_data_train = [fmri_filtered[i] for i in train_idx]  # Training subjects
print(np.array(fmri_data_train).shape)
fmri_train = Fmri_dataset(fmri_data_train, TR, block_duration)
train_loader = DataLoader(fmri_train, collate_fn=fmri_samples_collate_fn, **params)

# Test set
fmri_data_test = [fmri_filtered[i] for i in test_idx]
print(np.array(fmri_data_test).shape)
fmri_test = Fmri_dataset(fmri_data_test, TR, block_duration)
test_loader = DataLoader(
    fmri_test, collate_fn=fmri_samples_collate_fn, **params)

# ***************************************************************
# Inspect Accuracy results

# Block duration
print(f'Block duration = {block_duration}')

# Define model
model = FCN(n_regions, n_labels)  # time points == x, regions == rows
model = model.to(device)
print(model)
print("{} paramters to be trained in the model\n".format(count_parameters(model)))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()
num_epochs = 10
#adj_mat = 'a'

best_acc, best_confusion_matrix, best_predictions, best_target_classes, best_prop, best_count = model_fit_evaluate(
    model, adj_mat, device, train_loader, test_loader, n_labels, optimizer, loss_func, num_epochs)

# Save results to file
version = 1
with open('best_confusion_matrix{}.txt'.format(version), 'w') as f:
    for row in best_confusion_matrix:
        f.write(' '.join([str(a) for a in row]) + '\n')
