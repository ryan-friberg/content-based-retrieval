import os
from torch.utils.data import Dataset
import torch

'''
This file defines a dataset that is built after model training. It essentially serves
as the pre-compution for all the feature extractions in the dataset. The main purpose of 
this dataset is to speed up search by removing search-time model inference
'''

class SearchDataset(Dataset):
    def __init__(self, data_dir, model, associated_dataset, extract_features=True):
        self.data_dir = data_dir
        self.associated_dataset = associated_dataset

        if (not os.path.exists(data_dir)):
            os.mkdir(data_dir)

        if extract_features:
            self.extract_and_save_features(data_dir, model)
        
        self.files = self.get_filenames(self.data_dir)
        self.num_files = len(self.associated_dataset)

    def __getitem__(self, idx):
        return self.files[idx]

    def __len__(self):
        return self.num_files

    def extract_and_save_features(self, data_dir, model):
        # TODO: once training is finished, this function should be called
        # it should pre-compute and store the extracted features of each image to speed up search time
        # torch.save each tensor to file?
        model.eval()  # Set the model to evaluation mode

        for idx in range(len(self.associated_dataset)):
            data, _ = self.associated_dataset[idx]

            # Extract features
            with torch.no_grad():
                features = model(data.unsqueeze(0))  # Add batch dimension if necessary

            # Save the features
            feature_file = os.path.join(data_dir, f'features_{idx}.pt')
            torch.save(features, feature_file)

    def get_filenames(self):
        filenames = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        return filenames