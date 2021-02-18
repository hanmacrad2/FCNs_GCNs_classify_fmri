import numpy as np
import pickle
import pandas as pd
import glob
import os
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt 
from nilearn import plotting

import seaborn as sns

#Notes
#type python in terminal > then can run python code :D 


print('hi')

def get_fmri_data(root_pth, task_type):
    '''Get fmri data for a given task ''' 
    fmri_total = []
    subj_list = []
    for f in glob.glob(os.path.join(root_pth, task_type)+'/*.txt'):
        subj = f.split('/')[-1].split('_')[0]
        subj_list.append(subj)
        filename = os.path.join(root_pth, task_type)+'/'+ subj +'_schaefer_400ROI_'+ task_type.lower()+'.txt'
        #print(filename)
        ts_df = np.loadtxt(filename)
        fmri_total.append(ts_df)
    
    return fmri_total, subj_list

def preprocess_fmri(fmri):
    '''Preprocess '''

    return fmri

def get_rsfmri_adj_matrix(root_pth):
    '''Get resting state & return adjacency matrix '''
    
    #Connectivity
    correlation_measure = ConnectivityMeasure(kind='correlation')
    #Resting state fmri 
    rest_task = 'Rest'
    corr_rs_fmri = []

    #Resting state files 
    for f in glob.glob(os.path.join(root_pth, rest_task)+'/*.txt'):
        subj = f.split('/')[-1].split('_')[0]
        filename = os.path.join(root_pth, rest_task) +'/'+ subj +'_schaefer_400ROI_'+ rest_task.lower() +'.txt'
        #print(filename)
        ts_df = np.loadtxt(filename)
        correlation_matrix = correlation_measure.fit_transform([ts_df])[0]
        corr_rs_fmri.append(correlation_matrix)

    adj_matrix = np.array(corr_rs_fmri).mean(axis=0) #Is it the first access - yes if size 644 x... 

    return adj_matrix

def plot_corr_matrix(correlation_matrix):

    plt.figure(1)
    x = np.linspace(0, 20, 100)
    plt.plot(x, np.sin(x))
    plt.show()

    hmap = sns.heatmap(correlation_matrix)
    print('?')
    #plt.show()

    # Mask out the major diagonal
    #np.fill_diagonal(correlation_matrix, 0)
    #plotting.plot_matrix(correlation_matrix, colorbar=True,
    #                 vmax=0.8, vmin=-0.8)
    


def main():
    root_pth = '/camcan/schaefer_parc/'
    fmri_movie, subj_list = get_fmri_data(root_pth, 'Movie')
    print(np.array(fmri_movie).shape) #(193, 400)
    print(len(subj_list)) #644 subjects

    rs_fmri, subj_list_rs = get_fmri_data(root_pth, 'Rest')
    print(rs_fmri) #Size (261, 400)
    print(len(subj_list_rs)) #644 subjects

    #Adjacency matrix 
    adj_mat = get_rsfmri_adj_matrix(root_pth)
    plot_corr_matrix(adj_mat)

    #Other
    print(torch.__version__)

if __name__ == "__main__":
    main()