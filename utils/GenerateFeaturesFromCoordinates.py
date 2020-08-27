import pandas as pd
import numpy as np
from pyproj import Geod

class GenerateFeaturesFromCoordinates:

    def __init__(self, coordinates_path, bs_info):
        self.coordinates_path = coordinates_path
        self.bs_info = bs_info

        # Load csv of positions
        self.coordinates = self._load_coordinates()

        # TODO: Get speed
        self.speed = 30

        # TODO: ASsert bs_info
        # 
        # Generate features 
        self.features_df = self._generate_features()

    def get_features_df(self):
        return self.features_df


    def _load_coordinates(self):

        # Open .csv file and read
        # 
        # Return dataframe with lat, lon
        return pd.read_csv(self.coordinates_path)


    def _generate_feature_df(self):
        # Create matrix with neessary columns.
        #
        # Return as df
        feature_df = pd.DataFrame(columns=['index','lon','lat','speed','distance','delta_lon','delta_lat', 'cell_lon', 'cell_lat', 'cell_freq'])
        return feature_df

    
    def _populate_positions(self, df):
        # Populate dataframe with position coordinates
        #
        # Returns manipulated dataframe
        df['index'] = self.coordinates['index']
        df['lat'] = self.coordinates['lat']
        df['lon'] = self.coordinates['lon']
        return df

    def _populate_cell_features(self, df):
        # Compulate df with cell specific features
        #
        # Returns dataframe

        # Add frequency
        df['cell_freq'] = self.bs_info['cell_freq']

        # Add Position
        df['cell_lat'] = self.bs_info['cell_lat']
        df['cell_lon'] = self.bs_info['cell_lon']
 
        return df
        

    def _compute_distance(self, df):
        # Compute distance and return distance between all positions and cell position.
        wgs84_geod = Geod(ellps='WGS84')
        distance = np.zeros((df.shape[0],))

        for ii in np.arange(df.shape[0]):
            cell_lon = df.loc[ii, 'cell_lon']
            cell_lat = df.loc[ii,'cell_lat']
            meas_lon =  df.loc[ii,'lon']
            meas_lat =  df.loc[ii,'lat']
            _, _, dist = wgs84_geod.inv(cell_lon, cell_lat, meas_lon, meas_lat) 
            distance[ii] = dist / 1000 # Convert to km

        return distance

    def _compute_delta_positions(self,df):
        # Compute delta coordiantes between measurement coordinates and cell position
        delta_lat = df['lat'] - df['cell_lat']
        delta_lon = df['lon'] - df['cell_lon']

        return delta_lat, delta_lon



    def _generate_features(self):

        # Create feature df
        feature_df = self._generate_feature_df()

        # Populate from position df
        feature_df = self._populate_positions(feature_df)

        # Add frequency and cell position
        feature_df = self._populate_cell_features(feature_df)

        # Compute distance, add to df
        feature_df['distance'] = self._compute_distance(feature_df) 
        
        # Compute delta_lat, delta_lon, add to df
        feature_df['delta_lat'], feature_df['delta_lon'] = self._compute_delta_positions(feature_df)

        # Add speed
        feature_df['speed'] = self.speed

        # Return dataframe
        return feature_df