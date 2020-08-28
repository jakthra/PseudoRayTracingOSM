from utils import GenerateFeaturesFromCoordinates, DriveTestPredictionSet
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from easydict import EasyDict as edict
from model import SkynetModel
from experimentlogger import load_experiment
import torch

# Load new positions and generate features
bs_info_103068 = {'cell_lat': 51.489010, 'cell_lon': 7.403921, 'cell_freq': 1800}
features_103068 = GenerateFeaturesFromCoordinates('dataset\\103068.csv', bs_info_103068).get_features_df().drop(['cell_lat', 'cell_lon','lat','lon'],axis=1)

bs_info_114809 = {'cell_lat': 51.492758, 'cell_lon': 7.412209, 'cell_freq': 1800}
features_114809 = GenerateFeaturesFromCoordinates('dataset\\103068.csv', bs_info_114809).get_features_df().drop(['cell_lat', 'cell_lon','lat','lon'],axis=1)

# Load input and target scalers
dortmund_dataset_names = ['campus', 'urban', 'suburban', 'highway']
mnos = ['tmobile', 'o2', 'vodafone']
dortmund_features = []
dortmund_targets = []

for dortmund_data in dortmund_dataset_names:
    mno_features = [] 
    mno_targets = [] 
    for mno in mnos:
        dortmund_input_df = pd.read_pickle('dataset\\processed\\dortmund_feature_{}_{}.pkl'.format(mno, dortmund_data)).drop(['alt', 'cell_lat', 'cell_lon','lat','lon'],axis=1)
        dortmund_index = dortmund_input_df['index']
        dortmund_input_df.drop(['index'], axis=1, inplace=True)
        mno_features.append(dortmund_input_df)

        dortmund_target_df = pd.read_pickle('dataset\\processed\\dortmund_output_{}_{}.pkl'.format(mno, dortmund_data)).drop(['sinr', 'rsrq'],axis=1)
        mno_targets.append(dortmund_target_df)

    dortmund_features.append(pd.concat(mno_features)) 
    dortmund_targets.append(pd.concat(mno_targets)) 

all_german_features = pd.concat(dortmund_features)
all_german_targets = pd.concat(dortmund_targets)

input_scaler = StandardScaler().fit(all_german_features)
target_scaler = StandardScaler().fit(all_german_targets)

# Create pytorch dataset
dataset_103068 = DriveTestPredictionSet(features_103068.drop('index',axis=1), features_103068['index'].to_numpy(),"dataset/images/103068_png/", input_scaler)
dataset_114809 = DriveTestPredictionSet(features_114809.drop('index',axis=1), features_114809['index'].to_numpy(),"dataset/images/114809_png/", input_scaler)

# load model
exp_root_path = "exps/"
model_id = "6823d387-c934-40eb-81f3-afd34baa962d"
exp = load_experiment(model_id, root_path = exp_root_path)
args = edict(exp.config)
args.is_cuda = False
model = SkynetModel(args, target_scaler)
model.load_state_dict(torch.load('exps/models/6823d387-c934-40eb-81f3-afd34baa962d_model_0.748.pt', map_location=torch.device('cpu')))
model.eval()

# Predict using stored model
def evaluate_dataset(dataset_loader):
    with torch.no_grad():
        MSE = np.zeros((len(dataset_loader),))
        RMSE = np.zeros((len(dataset_loader),))
        for idx, (feature, image, target, dist_freq_offset) in enumerate(dataset_loader):
            dist = dist_freq_offset[0]
            freq = dist_freq_offset[1]
            offset = dist_freq_offset[2]
            correction_, sum_output_ = model(feature, image, dist, freq, offset)
            print(sum_output_)
            #P  = model.predict_physicals_model(feature, dist, freq, offset)
            
            #unnorm_predicted = scaler.inverse_transform(sum_output_.cpu().numpy())

            try:
                sum_output = torch.cat([sum_output, sum_output_],0)
            except:
                sum_output = sum_output_
    return sum_output

test_loader = torch.utils.data.DataLoader(dataset_103068, batch_size=1, drop_last=False, shuffle=False) 
sum_output = evaluate_dataset(test_loader)
unnorm_predicted = target_scaler.inverse_transform(sum_output.cpu().numpy())

np.savetxt('predictions/103068_rsrp.csv',unnorm_predicted, delimiter=',')

test_loader = torch.utils.data.DataLoader(dataset_114809, batch_size=1, drop_last=False, shuffle=False) 
sum_output = evaluate_dataset(test_loader)
unnorm_predicted = target_scaler.inverse_transform(sum_output.cpu().numpy())

np.savetxt('predictions/114809_rsrp.csv',unnorm_predicted, delimiter=',')



    
    