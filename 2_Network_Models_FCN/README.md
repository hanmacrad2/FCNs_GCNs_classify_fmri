## Classification of Network parcellated fmri data

## Overview

This invloved parcellating the fmri data into distinctive networks in the script __1_Networks_Data.py__. The parcellation was based on the 400 parcel parcellation matched to [Yeo 7 Network Parcellation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal). 
The FCN model was ten applied to the fmri data of each parcellated network separately to determine the indiviudal predictive power of each of the 7 parcellated networks across individuals in the script __2_Network_Models.py__

## Code

####  ```__1_Networks_Data.py__```
- Parcellates the fmri data into distinctive networks in the script

#### ```2_Network_Models.py```

- Contains the class ```Network_Model()``` which involved training the model based on the parcellated data of each of the 7 networks including
- 
  - The method ```create_network_data(self)``` which
    - Adds a column ```network``` to ```df_network``` which specifies the full name of the metric inferred from an abbreviation in the [Yeo parcellation file](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_7Networks_order.txt)
   - The method ```get_df_results_networks(self)``` which
      - Runs the FCN model using each network data indiviually

## Results 
The Visual Network had the highest classification accuracy across subjects. That is the model was able to best detect an underlying temporal trend across the fmri data pertaining to the Visual network. It had a classification accuracy of 71.4% as shown in the table below. THis is followed by the Somatomotor Network which has a test accuracy of 62.0%. In the first plot, the model accuracy across each of the different time blocks is shown, to get an idea of how the classifiers work across the duration of the movie. 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network_model_results_table.PNG" width="400" />

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network-model-plot1.png" width="600" />

The model training was repeated for 10 runs so that error bars of the standard deviation could be displayed as below, as well as the mean, to get a sense of the consistency of each of the network/model results. 

<img src="https://github.com/hanmacrad2/FCNs_GCNs_classify_fmri/blob/master/2_Network_Models_FCN/Results/network-model-stats.png" width="600" />
