import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from models.feature_extractor import VAE, SimpleCNN, ViTFeatureExtractor


# Step 1: train VAE

dataset_path = "./datasets/galaxies"
batch_size = 32
epochs = 10

# Initialize VAE
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Data loading and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Training function
def train_vae(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch, _ in dataloader:
            optimizer.zero_grad()
            images = batch
            reconstructed, mu, logvar = model(images)
            recon_loss = F.mse_loss(reconstructed, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

train_vae(vae, dataloader, optimizer, epochs)

# step 2: similarity scoring to find similar images
def extract_features(image, model):
    model.eval()
    with torch.no_grad():
        features, _, _ = model(image)
    return features

def calculate_similarity(features1, features2, w_cosine=0.33, w_l1=0.33, w_l2=0.33):
    """
    Compute a combined similarity score using cosine similarity, L1, and L2 distances.
    - features1: Tensor of features from the first set of images.
    - features2: Tensor of features from the second set of images.
    For now we just have equal weighting but it would make sense to try out different weighting
    """
    cosine_sim = F.cosine_similarity(features1, features2, dim=1)
    l1_dist = torch.norm(features1 - features2, p=1, dim=1)
    l2_dist = torch.norm(features1 - features2, p=2, dim=1)

    l1_dist = 1 - torch.sigmoid(l1_dist)
    l2_dist = 1 - torch.sigmoid(l2_dist)

    return w_cosine * cosine_sim + w_l1 * l1_dist + w_l2 * l2_dist

# after the training we extract the features from the dataset
def extract_features_from_dataset(dataloader, model):
    features = []
    with torch.no_grad():
        for batch, _ in dataloader:
            images = batch
            encoded = extract_features(images, model)
            features.append(encoded)
    return torch.vstack(features)

all_features = extract_features_from_dataset(dataloader, vae)

# STEP 3: find the similar images
image_to_compare_path = "./datasets/galaxies/Whirlpool_0.jpg"
query_image = transform(Image.open(image_to_compare_path).convert("RGB")).unsqueeze(0)
query_feature = extract_features(query_image, vae)

similarities = [calculate_similarity(query_feature, feature) for feature in all_features]

# Number of images we want to retieve
top_k = 5

# Sort the similarities and get the indices of top matches
top_indices = np.argsort(similarities)[-top_k:][::-1]

# Retrieve and display the top similar images
for idx in top_indices:
    similar_image_path = dataset.imgs[idx][0]
    similar_image = Image.open(similar_image_path)
    similar_image.show() #we display the images one by one