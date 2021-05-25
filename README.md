## Fully Connected & Graphical Neural Networks to classify fmri movie data 
Application of Fully Connected Neural Networks (FCNs) & Graphical Convolutional Neural Networks (GCNs) to fmri movie data. Brain decoding

- [Overview](#overview)
- [Code](#Code)
  * [Sub-heading](#sub-heading)
    + [Sub-sub-heading](#sub-sub-heading)]
 


> This is a fixture to test heading levels

<!-- toc -->

## Overview

FCNs and GCNs (1st order, 5th order, 8th order) were used to classify time blocks of fmri data across subjects.
The fmri data came from the [Camcan study](https://www.cam-can.org/) and was recorded while subjects watched a Hitchcock movie
It transpired that the FCNs yielded a better predictive performance. FCNs were therefore used in the second part of the study for further analysis. This involved parcellating the fmri data into 7 key networks (based on the Scaehfer parcellation) and determining the classification power of each network data separately. 

## Introduction 
A central goal in neuroscience is to understand the mechanisms in the brain responsible for cognitive functions. A recent approach known as “brain decoding ”, involves inferring certain experimental points in time using pattern classification of brain activity across participants.

Here a multidomain brain decoder was proposed that automatically learns the spatiotemporal dynamics of brain response within a short time window using a deep learning approach. The decoding model was evaluated on the fmri recorded from 644 participants from the camcan study recorded while they watched a 7 minute Hitchcock film.
A number of Using a 10s window of fMRI response, the 21 cognitive states were identified with a test accuracy of 90% (chance level 4.8%). Performance remained good when using a 6s window (82%).

Main Focus 
- I. FCNs vs GCNs
- II. FCNs - Network parcellation

## Code

#### I. FCNs vs GCNs
- As in 1_FCNs_vs_GCNs_fmri_classification

#### II. FCN Network Parcellation

- As in 2_Network_Models_FCN

#### Util funcs
- fmri data
- 
#### Models
- model_fcn_gcn
- model_fcn

## Results 

#### FCN > GCN across all tests
Show:

The following results for comparing the FCN model to the GCN model 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/1_FCNs_vs_GCNs_fmri_classification/miscellaneous_results/model_results.PNG" width="700" />



| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |




#### Separating out network data;
Visual had the best 

## Slides
- Intro
- FCNs vs GCNs
- Network parcellation

## Using the Resource
#### Prequisites
- The package requirements are written in the file requirements.text
- These can me installed by typing the following in your terminal;

pip install -r requirements.txt 

#### Insallation 
Open a terminal and type:
git clone https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri.git




## References 
