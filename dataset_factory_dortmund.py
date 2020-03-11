from torch.utils.data import Dataset
from torchvision import transforms
from numpy import genfromtxt
import numpy as np
from utils.tools import gps_distance
from pyproj import Geod
import pandas as pd

def generate_feature_output_matrix(raw_csv):
    # Assume csv file consists of #index,lon,lat,speed,sinr,rsrp,rsrq,cellLon,cellLat

    ## Feature matrix
    # distance with pyproj
    wgs84_geod = Geod(ellps='WGS84')
    distance = np.zeros((raw_csv.shape[0],))

    for ii in np.arange(raw_csv.shape[0]):
        cell_lon = raw_csv[ii, 7]
        cell_lat = raw_csv[ii, 8]
        meas_lon =  raw_csv[ii, 1]
        meas_lat =  raw_csv[ii, 2]
        _, _, dist = wgs84_geod.inv(cell_lon, cell_lat, meas_lon, meas_lat)
        distance[ii] = dist

    # Delta lat, lon
    delta_lat = raw_csv[:,2] - raw_csv[:, 8]
    delta_lon = raw_csv[:,1] - raw_csv[:, 7]

    feature_matrix = pd.DataFrame()
    feature_matrix['lon'] = raw_csv[:, 1]
    feature_matrix['lat'] = raw_csv[:, 2]
    feature_matrix['speed'] = raw_csv[:, 3]
    feature_matrix['distance'] = distance
    feature_matrix['delta_lon'] = delta_lon
    feature_matrix['delta_lat'] = delta_lat
    feature_matrix['cell_lon'] = raw_csv[:, 7]
    feature_matrix['cell_lat'] =  raw_csv[:, 8]

    ## Output matrix
    output_matrix = pd.DataFrame()
    output_matrix['sinr'] = raw_csv[:, 4]
    output_matrix['rsrp'] = raw_csv[:, 5]
    output_matrix['rsrq'] = raw_csv[:, 6]

    # Remove nan
    feature_matrix.dropna(inplace=True)
    output_matrix.dropna(inplace=True)


    return feature_matrix, output_matrix



def dataset_factory_dortmund(use_images=True, image_folder="images/snap_dk_250_png", transform=True, data_augment_angle=20):
    #index,lon,lat,speed,sinr,rsrp,rsrq,cellLon,cellLat

    #Longitude,Latitude,Speed,Distance,Distance_x,Distance_y,PCI_64,PCI_65,PCI_302	

    # Missing Distance, distance_x and distance_y



    tmobile_urban = genfromtxt('dataset\\dortmund\\tmobile_urban.csv', delimiter=',')
    feature_matrix, output_matrix = generate_feature_output_matrix(tmobile_urban)
    print(feature_matrix.head())
    print(output_matrix.head())




if __name__ == '__main__':
    dataset_factory_dortmund()