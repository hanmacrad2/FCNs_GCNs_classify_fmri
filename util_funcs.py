#Imports
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import preprocessing
import nilearn.signal
from nilearn.connectome import ConnectivityMeasure
import torch
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_add


#***********
#1. Get fmri data

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
    #Return 
    fmri_total = np.array(fmri_total)
    
    return fmri_total, subj_list

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


#********************************************************************************************************************

class Fmri_dataset(Dataset):
    
    def __init__(self, fmri_data_matrix, block_duration): #transform=False
        
        super(Fmri_dataset, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.subject_num = self.fmri_data_matrix.shape[0]
        self.block_duration = block_duration

    #Functions needed to generate data in parallel 
    def __len__(self):
        return self.subject_num

    def __getitem__(self, idx):

        #Data
        fmri_idx = self.fmri_data_matrix[idx]
        fmri_data, labels = self.split_fmri_blocks(fmri_idx, self.TR, self.block_duration)        
        tensor_x = torch.FloatTensor(fmri_data)
        tensor_y = torch.stack([torch.LongTensor([labels[ii]]) for ii in range(len(labels))])

        print('TENSOR x shape = {}'.format(tensor_x.size()))
        print('TENSOR y shape = {}'.format(tensor_y.size()))
        
        return tensor_x, tensor_y

    def split_fmri_blocks(self, fmri_idx, block_duration):
        
        '''Split fmri data into blocks of size block_duration, resulting in num_blocks blocks (total_time/block_duration)
        
        Returns:
        fmri data split into equal sized blocks of size num_blocks x block_duration x ROIs
        labels for corresponding blocks '''

        #i.Block params
        total_time = fmri_idx.shape[0]
        num_blocks = total_time // block_duration
        total_time = block_duration*num_blocks #Rounded number 
        fmri_idx = fmri_idx[:total_time, :] #Remove the last element; uneven number 
        
        #Labels of blocks
        labels = np.arange(0, num_blocks) 
        label_data = np.repeat(labels, block_duration)
        
        #Block change
        block_change = np.diff(label_data)
        idx_block_change = np.where(block_change)[0] + 1
        #Split into blocks
        blocks_labels = np.array(np.split(label_data, idx_block_change))
        
        #ii. fmri data
        fmri_blocks = np.array(np.array_split(fmri_idx, idx_block_change, axis=0))
        #Normalise data (each trial)
        fmri_data = []

        for t in np.arange(0,fmri_blocks.shape[0]):
            fmri_block_t = fmri_blocks[t][:,:] #ti == block, all time points up to num used, All ROIs   
            fmri_block_t = fmri_block_t.T
            fmri_data.append(fmri_block_t)
            
            #Filter #Normalise 
            #fmri_block_t = nilearn.signal.clean(fmri_block_t.T, sessions= None, detrend=True, 
                                       #standardize='zscore', confounds=None, low_pass= 0.1, 
                                       #high_pass= 0.01, t_r = TR, ensure_finite=False)
            
        #Array of data
        fmri_data = np.array(fmri_data)

        print('fmri data format shape = {}'.format(fmri_data))
        print('labels = {}'.format(labels.shape))

        return fmri_data, labels

def fmri_samples_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, targets = zip(*data) #iterator generates a series of tuples containing elements from each iterable
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.FloatTensor(torch.cat(images)) #.permute(0, 2, 1)
    targets = torch.LongTensor(torch.cat(targets).squeeze())
    
    return images, targets

def sparse_dense_mat_mul(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out

def check():
    print('Woopa')