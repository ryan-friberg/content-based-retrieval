from sklearn.cluster import KMeans
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from models.feature_extractor import VAE, SimpleCNN, ViTFeatureExtractor
from data.skydataset import SkyDataset
from PIL import Image
import numpy as np
import os
import torch.nn as nn
from collections import defaultdict
import torch
import torch.optim as optim
from tqdm import tqdm


"""
The steps here are:
1. Feature Extraction using VAE
2. Then KMeans Clustering: groups similar feature vectors together
    Each cluster is assigned a unique identifier/ Not traditional labels but more identifiers
    indicating which cluster a particular image belongs to.
3. Generating Triplets: 
    - the above "labels" determine which images should be paired together/are similar
    - determines which are in the same "cluster" and which arent
4. Training: model learns to differentiate between umages fron the same cluster and fron different clusters.
"""
# Path to your dataset
dataset_path = "./datasets/galaxies"

# Transformations for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


batch_size=128

dataset = SkyDataset(dataset_path, transform=preprocess)
dataloader = DataLoader(dataset, batch_size, shuffle=False)

### STEP 1: Feature extraction
feature_extractor = VAE()
feature_extractor.eval()

# Function to extract features
def extract_features(dataloader, feature_extractor):
    features = []
    image_paths = []
    with torch.no_grad():
        for inputs, paths in dataloader:
            outputs = feature_extractor(inputs)
            features.append(outputs)
            image_paths.extend(paths)
    return torch.cat(features).numpy(), image_paths

# Extracting features
features, image_paths = extract_features(dataloader, feature_extractor)

### STEP 2: Unsupervised clustering algorithm, K-means, to group the images.
# Clustering
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

### STEP 3: Generating Triplets and triplet loss
class TripletDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.labels_to_indices = self.map_labels_to_indices()

    def map_labels_to_indices(self):
   
        labels_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            labels_to_indices[label].append(idx)
        return labels_to_indices

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        positive_idx = np.random.choice(self.labels_to_indices[anchor_label])
        negative_label = np.random.choice(list(set(self.labels) - set([anchor_label])))
        negative_idx = np.random.choice(self.labels_to_indices[negative_label])

        positive_path = self.image_paths[positive_idx]
        negative_path = self.image_paths[negative_idx]

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.image_paths)
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
# Create Triplet Dataset
triplet_dataset = TripletDataset(image_paths, kmeans.labels_, transform=preprocess)
triplet_dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

### STEP 4: Model and Training Loop

# TODO: what is my model in unsupervised learning i am confused?
model = ViTFeatureExtractor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
triplet_loss = TripletLoss(margin=1.0)

# Training Loo

def train(model, dataloader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (anchor, positive, negative) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


train(model, triplet_dataloader, triplet_loss, optimizer, epochs=10)