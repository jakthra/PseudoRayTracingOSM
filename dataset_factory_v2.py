from utils import GenerateFeaturesFromCoordinates

bs_info_103068 = {'cell_lat': 51.489010, 'cell_lon': 7.403921, 'cell_freq': 1800}
features_103068 = GenerateFeaturesFromCoordinates   ('dataset\\103068.csv', bs_info_103068)
print(features_103068.get_features_df())

bs_info_114809 = {'cell_lat': 51.492758, 'cell_lon': 7.412209, 'cell_freq': 1800}
features_114809 = GenerateFeaturesFromCoordinates   ('dataset\\103068.csv', bs_info_114809)
print(features_114809.get_features_df())

    
    