from torch.utils.data import Dataset
from torchvision import transforms
from numpy import genfromtxt
import numpy as np
from pyproj import Geod
import pandas as pd
import glob
import os
import torch
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.invert import Invert
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize 
from pathloss_38901 import pathloss_38901
import matplotlib.pyplot as plt

def generate_feature_output_matrix_dortmund(raw_csv):
    # Assume csv file consists of #index,lon,lat,speed,sinr,rsrp,rsrq,cellLon,cellLat

    ## Feature matrix
    # distance with pyproj
    wgs84_geod = Geod(ellps='WGS84')
    distance = np.zeros((raw_csv.shape[0],))

    for ii in np.arange(raw_csv.shape[0]):
        cell_lon = raw_csv.loc[ii, 'cellLon']
        cell_lat = raw_csv.loc[ii,'cellLat']
        meas_lon =  raw_csv.loc[ii,'lon']
        meas_lat =  raw_csv.loc[ii,'lat']
        _, _, dist = wgs84_geod.inv(cell_lon, cell_lat, meas_lon, meas_lat) 
        distance[ii] = dist / 1000 # Convert to km

    # Delta lat, lon
    delta_lat = raw_csv['lat'] -  raw_csv['cellLat']
    delta_lon = raw_csv['lon'] - raw_csv['cellLon']

    feature_matrix = pd.DataFrame()
    feature_matrix['lon'] = raw_csv['lon']
    feature_matrix['lat'] = raw_csv['lat']
    feature_matrix['speed'] = raw_csv['speed']
    feature_matrix['alt'] = raw_csv['alt']
    feature_matrix['distance'] = distance
    feature_matrix['delta_lon'] = delta_lon
    feature_matrix['delta_lat'] = delta_lat
    feature_matrix['cell_lon'] = raw_csv['cellLon']
    feature_matrix['cell_lat'] =  raw_csv['cellLat']
    feature_matrix['cell_freq'] = raw_csv['cellFreq']
    ## Output matrix
    output_matrix = pd.DataFrame()
    output_matrix['sinr'] = raw_csv['sinr']
    output_matrix['rsrp'] = raw_csv['rsrp']
    output_matrix['rsrq'] = raw_csv['rsrq']

    # Remove nan
    feature_matrix.dropna(inplace=True)
    output_matrix.dropna(inplace=True)


    return feature_matrix, output_matrix

def process_dtu_data(path):
    dtu = pd.read_csv(path)

    # Process dtu dataframe such that dortmund dataset and denmark dataset provide the same input features
    new_dtu = pd.DataFrame()
    new_dtu['lon'] = dtu['Longitude'] 
    new_dtu['lat'] = dtu['Latitude'] 
    new_dtu['speed'] = dtu['Speed'] 
    new_dtu['distance'] = dtu['Distance']
    new_dtu['delta_lon'] = dtu['Distance_x'] 
    new_dtu['delta_lat'] = dtu['Distance_y']  
    new_dtu['cell_lat'] = 55.7846702
    new_dtu['cell_lon'] = 12.523481
    new_dtu['cell_freq'] = None

    idx_64 = np.where(dtu['PCI'] == 64)
    idx_65 = np.where(dtu['PCI'] == 65)
    idx_302 = np.where(dtu['PCI'] == 302)
    new_dtu.loc[idx_64[0], 'cell_freq'] = 811
    new_dtu.loc[idx_65[0], 'cell_freq'] = 811
    new_dtu.loc[idx_302[0], 'cell_freq'] = 2630
        

    return new_dtu

def link_budget(estimated_rx_losses, features, estimated_tx_power):
    
    # Find estimated rx losses using a theoretical path loss model
    path_loss = pathloss_38901(features['distance']*1000, features['cell_freq']/1000)
        
    # Link budget
    P_rx = estimated_tx_power - path_loss + estimated_rx_losses

    # RSRP conversion
    P_rsrp = P_rx - 10*np.log10(12*100)

    return P_rsrp

def calibrate_link_budget(estimated_rx_losses, features, measurements, estimated_tx_power):

    P_rsrp = link_budget(estimated_rx_losses, features, estimated_tx_power)

    mse = mean_squared_error(P_rsrp, measurements['rsrp'])

    return mse


def optimize_link_budget(features, measurements):
    res = minimize(calibrate_link_budget, 0, args=(features, measurements, 46))
    return res



def process_datasets():
    # Process data from dortmund
    for file in glob.glob("dataset\\dortmund\\*.csv"):
        file_df = pd.read_csv(file)
        feature_matrix, output_matrix = generate_feature_output_matrix_dortmund(file_df)
        file_name_split = file.rsplit("\\")
        file_name = file_name_split[-1] # Get last element
        feature_matrix.to_pickle('dataset\\processed\\dortmund_feature_{}.pkl'.format(file_name[:-4]))
        output_matrix.to_pickle('dataset\\processed\\dortmund_output_{}.pkl'.format(file_name[:-4]))

    # Process DTU dataset
    dtu_df = process_dtu_data('dataset\\dtu\\feature_matrix.csv')
    dtu_df.to_pickle('dataset\\processed\\dtu_feature.pkl')
    dtu_output_df = pd.read_csv('dataset\\dtu\\output_matrix.csv', usecols=[1,2,3,4])
    dtu_output_df.rename(columns={"RSRP":'rsrp', "SINR":'sinr', "Power": 'rssi', 'RSRQ':'rsrq'}, inplace=True)
    dtu_output_df.to_pickle('dataset\\processed\\dtu_output.pkl')

    # Everything saved as .pkl

def dataset_factory(dortmund_images_path="dortmund_images\\campus_png\\tmobile\\", dtu_images_path="images\\snap_dk_250_64_64\\", data_augmentation_angle=20):

    # Training set to consist of entire DTU dataset
    # Test set to consist of entire Dortmund dataset

    # Load DTU dataset 
    dtu_input_df = pd.read_pickle('dataset\\processed\\dtu_feature.pkl').drop(['cell_lat', 'cell_lon','lat','lon'], axis=1)
    dtu_target_df = pd.read_pickle('dataset\\processed\\dtu_output.pkl').drop(['sinr', 'rsrq', 'rssi'],axis=1)
    dtu_input_df.sort_index(axis=1, inplace=True)

    # Find offset
    dtu_offset = np.zeros((len(dtu_input_df), ))
    print(dtu_offset.shape)
    idx_811 = np.where(dtu_input_df['cell_freq'] == 811)
    offset_811 = optimize_link_budget(dtu_input_df.loc[idx_811], dtu_target_df.loc[idx_811])
    dtu_offset[idx_811] = offset_811.x

    idx_2630 = np.where(dtu_input_df['cell_freq'] == 2630)
    offset_2630 = optimize_link_budget(dtu_input_df.loc[idx_2630], dtu_target_df.loc[idx_2630])
    dtu_offset[idx_2630] = offset_2630.x

    print(dtu_input_df.head())
    print(dtu_target_df.head())


    # fig = plt.figure(figsize=(5,5))
    # before_optim = link_budget(0, dtu_input_df.loc[idx_811], 46)
    # after_optim = link_budget(offset_811.x, dtu_input_df.loc[idx_811], 46)

    # plt.plot(dtu_input_df.loc[idx_811[0],'distance'], before_optim, 'o', label='No optimization')
    # plt.plot(dtu_input_df.loc[idx_811[0],'distance'], after_optim, 'o', label='Optimizated')
    # plt.plot(dtu_input_df.loc[idx_811[0],'distance'], dtu_target_df.loc[idx_811[0], 'rsrp'], 'o', label='Measurements')
    # plt.legend()
    # plt.show()

    
    # fig = plt.figure(figsize=(5,5))
    # before_optim = link_budget(0, dtu_input_df.loc[idx_2630], 46)
    # after_optim = link_budget(offset_2630.x, dtu_input_df.loc[idx_2630], 46)

    # plt.plot(dtu_input_df.loc[idx_2630[0],'distance'], before_optim, 'o', label='No optimization')
    # plt.plot(dtu_input_df.loc[idx_2630[0],'distance'], after_optim, 'o', label='Optimizated')
    # plt.plot(dtu_input_df.loc[idx_2630[0],'distance'], dtu_target_df.loc[idx_2630[0], 'rsrp'], 'o', label='Measurements')
    # plt.legend()
    # plt.show()


    # Load Dortmund dataset
    dortmund_input_df = pd.read_pickle('dataset\\processed\\dortmund_feature_tmobile_campus.pkl').drop(['alt', 'cell_lat', 'cell_lon','lat','lon'],axis=1)
    dortmund_target_df = pd.read_pickle('dataset\\processed\\dortmund_output_tmobile_campus.pkl').drop(['sinr', 'rsrq'],axis=1)
    dortmund_input_df.sort_index(axis=1, inplace=True)

    
    dortmund_offset = np.zeros((len(dortmund_input_df), ))
    for group in dortmund_input_df.groupby(['cell_freq']):
        cell_freq = group[0]
        idx = np.where(dortmund_input_df['cell_freq'] == cell_freq)
        offset = optimize_link_budget(dortmund_input_df.loc[idx], dortmund_target_df.loc[idx])
        dortmund_offset[idx] = offset.x

    print(dortmund_input_df.head())
    print(dortmund_target_df.head())

    # Setup normalization using training data
    input_scaler = StandardScaler().fit(dtu_input_df)
    target_scaler_dtu = StandardScaler().fit(dtu_target_df)
    target_scaler_dortmund = StandardScaler().fit(dortmund_target_df)

    # Setup image transformation
    composed = transforms.Compose([transforms.ToPILImage(),  transforms.Grayscale(), Invert(), transforms.RandomAffine(data_augmentation_angle, shear=10), transforms.ToTensor()])

    # Create generators for test and training data
    train_dataset = DrivetestDataset(dtu_input_df, dtu_target_df, dtu_images_path, composed, input_scaler, target_scaler_dtu, dtu_offset)
    test_dataset = DrivetestDataset(dortmund_input_df, dortmund_target_df, dortmund_images_path, composed, input_scaler, target_scaler_dortmund, dortmund_offset)

    # Test output of training and test generators

    # X, A, y, dist_and_freq = next(iter(training_dataset))
    # print(X.shape)
    # print(A.shape)
    # print(y.shape)
    # print(dist_and_freq)

    # X, A, y, dist_and_freq = next(iter(test_dataset))
    # print(X.shape)
    # print(A.shape)
    # print(y.shape)
    # print(dist_and_freq)

    return train_dataset, test_dataset

class DrivetestDataset(Dataset):
    def __init__(self, features, targets, image_folder, transform, input_scaler, target_scaler, offset):
        
        self.offset = offset

        self.targets = targets.to_numpy()
        self.features = features.to_numpy()
        self.distances = features['distance']*1000
        self.frequency = features['cell_freq']/1000
        
        self.image_folder = image_folder
        self.transform = transform
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.image_size = io.imread(os.path.join(self.image_folder, "{}.png".format(0))).shape

    def get_811Mhz_idx(self):
        return np.argwhere(np.asarray(self.features['cell_freq'] == 811))

    def get_2630Mhz_idx(self):
        return np.argwhere(np.asarray(self.features['cell_freq'] == 2630))

    def __getitem__(self, index):
        X_norm = self.input_scaler.transform(self.features[index].reshape(1, -1))
        X = torch.from_numpy(X_norm).float().squeeze()  # Features (normalized)
        img_name = os.path.join(self.image_folder, "{}.png".format(index))
        image = io.imread(img_name)
        image = image / 255
        A = torch.from_numpy(image).float().permute(2,0,1)
            
        y_norm = self.target_scaler.transform(self.targets[index].reshape(1, -1))
        y = torch.from_numpy(y_norm).float().squeeze(1) # Target

        # Path loss model requires additional information that is unnormalized, thus we keep these in seperate arrays that are non transformed and key dependent

        dist = torch.abs(torch.tensor(self.distances[index])).float().view(1)
        freq = torch.abs(torch.tensor(self.frequency[index])).float().view(1) 
        offset = torch.tensor(self.offset[index]).float().view(1) 


        A = self.transform(A)

        return X, A, y, [dist, freq, offset]

    def __len__(self):
        return len(self.features)



if __name__ == '__main__':
    
    process_datasets()
    dataset_factory()