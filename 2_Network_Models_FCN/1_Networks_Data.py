#Networks data
import pandas as pd
import numpy as np

df = pd.read_csv('networks_7_parcel_400.txt', delimiter = "\t")
df.columns = ['index', 'network_name', 'x3', 'x4', 'x5', 'x6']
list_networks = ['Vis', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Cont', 'Default']

#Network column
df['network'] = df['network_name']
#Subset networks 
for netw in list_networks:
    df.loc[df['network'].str.contains(netw), 'network'] = netw

#Extract from fmri
for netw in list_networks:
    indx = df.index[df["network"] == netw].tolist()
    

df.iloc[[7, 2, 3, 1, 6]]