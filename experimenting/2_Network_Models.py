# Application of FCNs & GCNs to fmri movie data (camcan study)
# Inspect accuracy results/confusion matrices across time

# Imports
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn import preprocessing, metrics, manifold
import warnings
import matplotlib.pyplot as plt
import importlib
import itertools
import scipy.io
import pandas as pd
import numpy as np
import datetime
import time
import math
import sys
import os

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

from models import *
from util_funcs import *
print('Start')
check()

# ****************************************
# Data
root_pth = '/camcan/schaefer_parc/'
task_type = 'Movie'
fmri, subj_list = get_fmri_data(root_pth, task_type)
print(np.array(fmri).shape)  # (193, 400)
print(len(subj_list))  # 644 subjects

#****************************************************
# Data preprocessing - filter + normalise fmri
def filter_fmri(fmri, standardize):
    'filter fmri signal'

    # fmri
    TR = 2.47
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

# *****************************************
# Network models
class Network_Model():
    
    '''Class to run FCN models for x7 network parcellation '''

    def __init__(self, fmri, network_file, block_duration):
        super(Network_Model, self).__init__()

        # Data
        self.fmri = fmri
        self.network_file = network_file
        self.list_networks = ['Vis', 'SomMot', 'DorsAttn','VentAttn', 'Limbic', 'Cont', 'Default']
        #self.df_network = self.create_network_data(self.network_file, self.list_networks)

        # fmri params
        self.TR = 2.47
        self.n_regions = np.array(fmri).shape[2]
        self.n_subjects = fmri.shape[0]
        self.block_duration = block_duration  # 8 #16 #6 8 -Factor of 192
        self.total_time = fmri.shape[1]
        self.n_blocks = self.total_time // block_duration
        self.n_labels = self.n_blocks

        # Model params
        self.n_model_repeats = 10
        self.device = torch.device("cpu")
        self.params = {'batch_size': 1,
                       'shuffle': True,
                       'num_workers': 2}
        self.test_size = 0.2
        self.num_train_sub = 0
        self.num_test_sub = 0
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

        indx_netw = self.df_network.index[self.df_network["network"] == networkX].tolist()
        fmri_network = self.fmri[:, :, indx_netw]
        print(f'{networkX} network fmri shape = {fmri_network.shape}')

        # Num regioins
        n_regions = fmri_network.shape[2]
        return fmri_network, n_regions

    def get_train_test_data(self, fmri_network):
        
        # Training/Test indices
        train_idx, test_idx = train_test_split(range(self.n_subjects), test_size= self.test_size, random_state= self.rs, shuffle=True)
        #Num subjects
        self.num_train_sub = len(train_idx)
        self.num_test_sub =  len(test_idx)
        print('Training on %d subjects, Testing on %d subjects' %
              (self.num_train_sub, self.num_test_sub))

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

        'Get dataframe of the network results'

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

        #Proportion -> Percentage
        percantage_correct = 100*(df_results['proportion']/self.num_test_sub)

        self.num_test_sub
        # Dataframe
        df_results['network'] = self.list_networks
        df_results['accuracy'] = list_acc
        df_results['proportion'] = list_prop

        #Format output
        df_results['proportion'] = df_results['proportion'].apply(lambda x: x.numpy())
        df_results['pcent'] = df_results['proportion'].apply(lambda x: 100*(x/self.num_test_sub))

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
            model, self.device, train_loader, test_loader, self.n_labels, optimizer, loss_func, num_epochs)

        return best_acc, best_prop

    
    def get_stats_network_results(self, dict_netw_results):
        'Aggregate mean stats across time points for a number of repitions'

        #Params
        df_results = pd.DataFrame()
        list_net = []; list_buckets = []; list_mean = []; list_std = []
        for net in dict_netw_results.keys():
            #print(net)
            l = list(iter(dict_netw_results[net].values()))
            for i in range(len(l[0])):
                #print(i)
                list_net.append(net)
                list_buckets.append(i)
                row_list = [row[i] for row in l] #get values column-wise
                #print(row_list)
                list_mean.append(np.mean(np.array(row_list)))
                list_std.append(np.std(np.array(row_list)))
        #Dataframe
        df_results['network'] = list_net
        df_results['time_bucket'] = list_buckets
        df_results['mean_pcent_fcn'] = list_mean
        df_results['std_pcent_fcn'] = list_std

        return df_results

    def repeat_model_stats(self):
        'Repeat model results + get mean + std across networks'

        df_network = self.create_network_data()
        dict_netw_results = {key: {} for key in self.list_networks}

        #Repeat to get std
        for i in np.arange(1, self.n_model_repeats):
            print('Iteration num: {}'.format(i))
            #Repeat for each network
            for networkX in self.list_networks:
                print(f'Network = {networkX}')
                fmri_networkX, n_regions = self.get_network_fmri(networkX)  # fmri or self.fmri??
                best_acc, best_prop = self.run_model(
                    fmri_networkX, n_regions)
                dict_netw_results[networkX][i] = 100*(best_prop/self.num_test_sub)
            
            df_stats_results = self.get_stats_network_results(dict_netw_results)

        return df_stats_results
        

#Network class
network_file = 'networks_7_parcel_400.txt'
block_duration = 6
netw_model = Network_Model(fmri_filtered, network_file, block_duration)
df_results = netw_model.get_df_results_networks()
#Redo to see tensor shapes
df_results2 = netw_model.get_df_results_networks()
index  = np.arange(0, len(df_results['proportion'][0]))
#Save
df_results.to_pickle('df_network_results.pkl')
df_results = pd.read_pickle('df_network_results.pkl')

#Plot proportions
def plot_model_stats(df_results, index): 

    'Plot mean/std of pcentage correct of subjects across time buckets'
    #Plot
    plt.figure(figsize=(10, 8))
    networks = ['Vis', 'SomMot', 'DorsAttn','VentAttn', 'Limbic', 'Cont', 'Default']
    colors = ["red", "blue" , "green", "orange", "purple", 'black', 'yellow']

    for network, color in zip(networks, colors):
        #Extract df network
        df_netX = df.loc[df['network'] == network] 
        print('df_netX.shape() = {}'.format(df_netX.shape()))
        plt.plot(index, df_results['mean_pcent_fcn'], '-o', yerr='std', color = colorX, label = df_results['network'][ind])
    
    plt.title(f'Network FCN models - Mean pcent of correctly classified subjects per time bucket (10 model runs)')
    plt.xlabel('Block')
    plt.ylabel('Mean % Subjects correct')
    plt.legend(loc="lower right", framealpha = 1)
    plt.show()

#**************************************************
#Plot of results
#Note - Plots of results ran in Results_plot.ipynb

#II Get mean/Std
df_model_stats = netw_model.repeat_model_stats()
df_model_stats.to_pickle('df_model_stats.pkl')

#Plot mean/std + error bars (See jupyter)
def plot_model_stats(df_results, index):

    'Plot mean/std of pcentage correct of subjects across time buckets'

    #Plot
    plt.figure(figsize=(10, 8))
    networks = ['Vis', 'SomMot', 'DorsAttn','VentAttn', 'Limbic', 'Cont', 'Default']
    colors = ["red", "blue" , "green", "orange", "purple", 'black', 'yellow']

    for network, color in zip(networks, colors):
        #Extract df network
        df_netX = df.loc[df['network'] == network] 
        print('df_netX.shape() = {}'.format(df_netX.shape())
        plt.plot(index, df_results['mean_pcent_fcn'], yerr='std', '-o', color = colorX, label = df_results['network'][ind])
    
    plt.title(f'Network FCN models - Mean pcent of correctly classified subjects per time bucket (10 model runs)')
    plt.xlabel('Block')
    plt.ylabel('Mean % Subjects correct')
    plt.legend(loc="lower right", framealpha = 1)
    plt.show()

#Run (Plotted in Jupyter)
plot_model_stats(df_results, index)

