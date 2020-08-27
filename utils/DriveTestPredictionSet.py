
import torch
from skimage import io, transform
from torchvision import transforms

class DriveTestPredictionSet(Dataset):
    def __init__(self, features, image_folder, input_scaler):
        
        try:
            self.index = index.to_numpy()
        except: 
            self.index = index
        self.features = features.to_numpy()
        self.distances = features['distance']*1000
        self.frequency = features['cell_freq']/1000

        
        self.image_folder = image_folder
        self.input_scaler = input_scaler
        self.image_size = io.imread(os.path.join(self.image_folder, "{}.png".format(0))).shape
        self.transform = transforms.Compose([transforms.ToPILImage(),  transforms.Grayscale(), Invert(), transforms.ToTensor()])

    def __getitem__(self, index):
        img_idx = self.index[index]
        X_norm = self.input_scaler.transform(self.features[index].reshape(1, -1))
        X = torch.from_numpy(X_norm).float().squeeze()  # Features (normalized)
        img_name = os.path.join(self.image_folder, "{}.png".format(img_idx))
        image = io.imread(img_name)
        image = image / 255
        A = torch.from_numpy(image).float().permute(2,0,1)
            
        # Path loss model requires additional information that is unnormalized, thus we keep these in seperate arrays that are non transformed and key dependent
        dist = torch.abs(torch.tensor(self.distances[index])).float().view(1)
        freq = torch.abs(torch.tensor(self.frequency[index])).float().view(1) 
        offset = torch.tensor(0).float().view(1) 


        A = self.transform(A)

        return X, A, y, [dist, freq, 0]

    def __len__(self):
        return len(self.index)