from utils import GenerateFeaturesFromCoordinates, DriveTestPredictionSet
import pandas as pd
import numpy as np

# Load new positions and generate features
bs_info_103068 = {'cell_lat': 51.489010, 'cell_lon': 7.403921, 'cell_freq': 1800}
features_103068 = GenerateFeaturesFromCoordinates('dataset\\103068.csv', bs_info_103068).get_features_df()

bs_info_114809 = {'cell_lat': 51.492758, 'cell_lon': 7.412209, 'cell_freq': 1800}
features_114809 = GenerateFeaturesFromCoordinates('dataset\\103068.csv', bs_info_114809).get_features_df()

# Load input and target scalers
# TODO: create scalers
dortmund_dataset_names = ['campus', 'urban', 'suburban', 'highway']
mnos = ['tmobile', 'o2', 'vodafone']
dortmund_datasets = []

for dortmund_data in dortmund_dataset_names:
    mno_datasets = {}
    mno_datasets_list = [] 
    for mno in mnos:
        dortmund_target_df = pd.read_pickle('dataset\\processed\\dortmund_output_{}_{}.pkl'.format(mno, dortmund_data)).drop(['sinr', 'rsrq'],axis=1)
        mno_datasets[mno] = dortmund_target_df
        mno_datasets_list.append(dortmund_target_df)
    dortmund_datasets.append(pd.concat(mno_datasets_list)) 

all_data_german = pd.concat(dortmund_datasets)

# Create pytorch dataset
dataset_103068 = DriveTestPredictionSet(features_103068, features_103068['index'].to_numpy(),"dataset/images/103068_png/", [])
dataset_114809 = DriveTestPredictionSet(features_114809, features_114809['index'].to_numpy(),"dataset/images/114809_png/", [])

# Predict using stored model

# Scale output values using target scalers

# Store prediction in .csv



    
    