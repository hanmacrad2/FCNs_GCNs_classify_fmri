## Fully Connected & Graphical Neural Networks to classify fmri movie data 
> Application of Fully Connected Neural Networks (FCNs) & Graphical Convolutional Neural Networks (GCNs) to fmri movie data. Brain decoding

- [Overview](#overview)
- [Introduction](#Introduction)
- [Results](#Results)
- [Code](#Code)
- [Using the Resource](#Code)
- [References](#Code)

## Overview

FCNs and GCNs (1st order, 5th order, 8th order) were used to classify time blocks of fmri data across subjects.
The fmri data came from the [Camcan study](https://www.cam-can.org/) and was recorded while subjects watched a Hitchcock movie
It transpired that the FCNs yielded a better predictive performance. FCNs were therefore used in the second part of the study for further analysis. This involved parcellating the fmri data into 7 key networks (based on the Scaehfer parcellation) and determining the classification power of each network data separately. 

## Introduction 
A central goal in neuroscience is to understand the mechanisms in the brain responsible for cognitive functions. A recent approach known as “brain decoding ”, involves inferring certain experimental points in time using pattern classification of brain activity across participants. Here a multidomain brain decoder was proposed that automatically learns the spatiotemporal dynamics of brain response within a short time window using a deep learning approach. The decoding model was evaluated on the fmri recorded from 644 participants from the camcan study recorded while they watched a 7 minute Hitchcock film.
A number of time windows for fmri classification were used including 6 blocks x TR, 8 blocks x TR and 16 blocks x TR, whereby TR is the repetition time (time it takes to scan all slices) and equal to 2.47 secs in this instance. As mentioned above, the main focus of the analyses was;

#### 1. Comparison of FCNs vs GCNs
The classification performance of the FCNs was compared to that of the GCNs to determine which was the optimal classifier in the context of decoding fmri data. 
As shown below, the best results were obtained using the FCN model, achieving a mean accuracy of 90.7%, and so the FCN was used for further analysis as below. 


#### 2. Network Model parcellation
This invloved parcellating the fmri data into distinctive cognitive networks, specifically 400 parcel parcellation matched to [Yeo 7 Network Parcellation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal). The 7 networks of cognitive function included the 
Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Default and Control Networks. The FCN model was applied to the fmri data of each parcellated network separately to determine the indiviudal predictive power of each of the 7 parcellated networks across all individuals fmri.

## Results 

#### FCN > GCN across all tests

The following accuracy results where obtained for the FCN and GCN models when used to classify the fmri data across timepoints. The FCN model performed significantly better then the GCN models. It obtained a mean accuracy of 90.7% compared to 78.5% for the 5th order GCN model, 75.1% for the 8th order GCN model and 54.1% for the 1st order GCN model. Thus for the future analyses the FCN model was used. It's architecture was also a lot more straightforward, a further advantage of the model. 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/1_FCNs_vs_GCNs_fmri_classification/miscellaneous_results/model_results.PNG" width="700" />


#### Network FCNs - Visual is best
The Visual Network had the highest classification accuracy across subjects. That is, the model was able to best detect an underlying temporal trend across the fmri data pertaining to the Visual network. It had a classification accuracy of 71.4% as shown in the table below. This was followed by the Somatomotor Network which has a test accuracy of 62.0%. In the first plot, the model accuracy for each of the different time blocks is shown, to get an idea of how the classifiers were performing across the duration of the movie. 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network_model_results_table.PNG" width="400" />

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network-model-plot1.png" width="600" />

The model training was repeated for 10 runs so that error bars of the standard deviation, as well as the mean, could be displayed as below to get a sense of the consistency of each of the network model results. 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network-model-stats.png" width="600" />

## Code

#### ``` 1_FCNs_vs_GCNs_fmri_classification```
- This folder contains the code for evaluating the FCNs and GCNs on the fmri data. The models are trained on various block durations and their performance metrics compared

#### ``` 2_Network_Models_FCN``` 
- This folder contains the code for parcellating the fmri data into distinctive cognitive networks. The FCN model was then applied to the fmri data of each parcellated network separately

#### ```util_funcs.py```
This script contains various utility functions related to loading the fmri data 
- ```get_fmri_data(root_pth, task_type)```
    - Loads the fmri data from the root path, where ```task_type = (Movie, Rest)```, pertaining to the fmri movie data or resting state data
    - The class ```Fmri_dataset(Dataset)``` which includes the function ```split_fmri_blocks()``` which; 
    - Splits the fmri data into blocks of size block_duration, resulting in num_blocks blocks (num_blocks = total_time/block_duration)
    - Thus the fmri is returned split into equal sized blocks of size num_blocks x block_duration x ROIs
- ```get_rsfmri_adj_matrix(root_pth)```
    - Gets the resting state data & returns the  adjacency matrix required for the graphical neural network models
  

#### Models
#### ```models_fcn.py```
- Contains the code for the Fully Connected Neural Network. 
    - The fmri data is averaged across it's time dimension before being inputed to the FCN, so that it goes from dimension Number of subjects x ROIs x Timepoints to dimension Number of subjects x ROIs x Timepoints
    - THe FCN architecture involved  two hidden layers, one of size hidden_dim and one of size hidden_dim/4. With dropout 0.5 for latter two sets of weights and dropout of self.dropout=0.2 for the input to hidden layer.

####  ```model_fcn_gcn.py```
- Contains the code for the Fully Connected Neural Network and Graphical Neural Network models (In the folder ``` 2_Network_Models_FCN```)

## Slides
- Intro
- FCNs vs GCNs
- Network parcellation

## Using the Resource
#### Prequisites
- The package requirements are written in the file requirements.text
- These can me installed by typing the following in your terminal;

```
pip install -r requirements.txt 
```

#### Installation 
Open a terminal and type:

```
git clone https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri.git
```



## References 

Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., ... & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of neurophysiology.

Zhang, Y., Tetrel, L., Thirion, B., & Bellec, P. (2021). Functional annotation of human cognitive states using deep graph convolution. NeuroImage, 231, 117847.

Zhang, Y., & Bellec, P. (2019). Functional annotation of human cognitive states using graph convolution networks.

