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
        self.n_blocks = total_time // block_duration
        self.n_labels = n_blocks

        # Model params
        self.model_repeats = 10
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

        indx_netw = self.df_network.index[df["network"] == networkX].tolist()
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
    
    def get_network_pcent(self, dict_netw_results, n_repeat):

        #Setup
        #list_acc = []; list_prop = []
        df_network = self.create_network_data()
        #df_results = pd.DataFrame()
        #dict_netw_results = {}

        for networkX in self.list_networks:
            print(f'Network = {networkX}')
            fmri_networkX, n_regions = self.get_network_fmri(networkX)  # fmri or self.fmri??
            best_acc, best_prop = self.run_model(
                fmri_networkX, n_regions)
            
            #Append percentage correct for each network
            dict_netw_results[networkX] = {n_repeat: 100*(best_prop/self.num_test_sub)}
            
            #list_acc.append(best_acc)
            #list_prop.append(best_prop)

        # Dictionary
        #dict_netw_results['network'] = self.list_networks
        #dict_netw_results['accuracy'] = list_acc
        #dict_netw_results['proportion'] = np.array(list_prop)
        #Format output
        #dict_netw_results['pcent'] = 100*(dict_netw_results/self.num_test_sub)

        return dict_netw_results


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
    
    def repeat_std(self):
        'Repeat model results + get std across networks'

        #Iteration #1
        repeat_num = 1
        dict_netw_results = {}
        dict_netw_results = get_network_pcent(dict_netw_results, repeat_num)
        count_repeats = 1

        #Repeat
        for i in np.arange(2, self.model_repeats):
            dict_netw_results = get_network_pcent(dict_netw_results, repeat_num)
            #dict_temp = get_network_pcent()
            #Add repeated results
            #dict_netw_results = {key: value + dict_temp[key] for key, value in dict_netw_results.items()}
            count_repeats += 1    

        #Get stats on results 
        for networkX in self.list_networks:
            l = list(iter(test_dict.values())) #Values
        
            d={}                                                                  #final ditionary
            for i in range(len(l[0])): 
            row_list = [row[i] for row in l]                     #get values column-wise
            d['location'+str(i+1)] = sum(row_list)/len(row_list)               #calculate avg




#Network class
network_file = 'networks_7_parcel_400.txt'
netw_model = Network_Model(fmri_filtered, network_file, block_duration)
df_results = netw_model.get_df_results_networks()
#Save
df_results.to_pickle('df_network_results.pkl')

#Plot proportions
def plot_network_acc_pcent(df_results):
    'Plot percentage of correctly classified subjects in each time bucket for all 7 networks'

    #Block index
    index  = np.arange(0, len(df_results['proportion'][0]))

    #Plot
    plt.figure(figsize=(10, 8))
    colors = ["red", "blue" , "green", "orange", "purple", 'black', 'yellow']
    #Plot
    for ind, colorX in enumerate(colors):
        percantage_correct = 100*((df_results['proportion'][ind])/129) 
        plt.plot(index, percantage_correct, '-o', color = colorX, label = df_results['network'][ind]) # marker = '*')
    
    plt.title(f'Network FCN models - Pcent of correctly classified subjects in each time bucket')
    plt.xlabel('Block')
    plt.ylabel('% Subjects correct')
    plt.legend(loc="lower right", framealpha = 1)
    plt.show()

#Results ran in Results_plot.ipynb
plot_network_acc_pcent(df_results)


