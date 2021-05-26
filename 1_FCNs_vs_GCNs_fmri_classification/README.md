## Fully Connected & Graphical Neural Networks to classify fmri movie data 

Application of Fully Connected Neural Networks (FCNs) & Graphical Convolutional Neural Networks (GCNs) to fmri movie data. 

## Overview

FCNs and GCNs (1st order, 5th order, 8th order), developed using pytorch, were used to classify time blocks of fmri data across subjects.
The decoding model was evaluated on the fmri recorded from 644 participants from the [Camcan study](https://www.cam-can.org/) recorded while they watched a 7 minute Hitchcock film. THe fmri data was of dimension 644 subjects x 193 timepoints x 400 Regions of Interest (ROIs). A number of time windows for fmri classification were used including 6, 8 and 16 blocks. Note that the repetition time (TR) of the fmri data, whci is the time it takes to scan all slices, is equal to 2.47 secs in this instance. The fmri data was split into equaly sized blocks of timepoints. 

The classification performance of the FCNs was compared to that of the GCNs to determine which was the optimal classifier in the context of decoding fmri data. The fmri data was split into equaly sized blocks of timepoints, for example, the 192 timepoints would be split into 26 x 8 blocks for a block duration of 8. THe FCN and GCN models were then used in a Machine Learning manner to classify each timepoint as coming from each of the 26 blocks. It transpired that the FCNs yielded a better predictive performance than the GCNs, achieving a mean accuracy of 90.7%, and so the FCNs were therefore used in the second part of the study for further analysis. 

## Code 

#### ```1_fmri_gcns_main.ipynb```
The main results were ran in this jupyter notebook. The models were run for block durations of 6, 8 and 16 to both filtered and filtered + normalised fmri data. The class methods and functionality are described in the ```REAMDME``` in [FCNs_GCNs_classify_fmri](https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/README.md%20%60%60%60)

## Results
The results obtained are as in the table below;

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/1_FCNs_vs_GCNs_fmri_classification/miscellaneous_results/model_results.PNG" width="700" />
