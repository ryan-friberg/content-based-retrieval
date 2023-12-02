import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm 
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
Defines the model-agnostic training and testing processes

TODO: it is a design decision about how many images in the batch we transform
      and how many transforms we apply at a time, we can do analyze what makes
      the most sense from a training time/ performance perspective. Having every
      image get into a pair effectively doubles the dataset size (which was
      already pretty massive) and this is on top of associating each color channel
      as a unique image in the transformer case. We will just need to be mindful
      of the training runtimes.
'''

# torchvision surprisingly does not have a built in noise-adding transform
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def generate_positive_pairs(batch, num_augmentations):
    # define a set of transformations that do not significantly alter
    # the visual content of the image (retrains visual features, positive associations)
    positive_transform_options = np.array([
        transforms.RandomHorizontalFlip(1.0),
        transforms.RandomVerticalFlip(1.0),
        transforms.RandomRotation(180),
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        AddGaussianNoise(0., 0.1)
    ])
    
    positive_pairs = []
    for i in range(batch.shape[0]):
        # randomly sample a set of transforms (to get diverse alterations throughout training)
        random_transforms = transforms.Compose(
            np.random.choice(positive_transform_options, size=num_augmentations, replace=False)
        )

        unaltered_image = batch[i]
        altered_image = random_transforms(unaltered_image)
        positive_pairs.append((unaltered_image, altered_image))
    
    return positive_pairs


def generate_negative_pairs(batch, num_augmentations):
    # define a set of transformations that significantly alter or even destroy
    # the visual content of the image (negative associations)

    # TODO: possibly make it chance to sample any other random image from the dataset
    # or transform current one if not. This should be viable since all of the galaxies should
    # all at least be subtly visually different, this could help the model learn more subtle
    # visual features

    negative_transform_options = np.array([
        transforms.RandomRotation(180),
        transforms.RandomPerspective(distortion_scale=0.85, p=1.0),
        transforms.RandomErasing(p=1.0, scale=(0.5, 0.5), ratio=(0.3, 3.3), value='random'),
        transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=1.0),
        transforms.GaussianBlur(kernel_size=8, sigma=(0.1, 2.0)),
        AddGaussianNoise(0., 2.0)
    ])

    negative_pairs = []
    for i in range(batch.shape[0]):
        # randomly sample a set of transforms (to get diverse alterations throughout training)
        random_transforms = transforms.Compose(
            np.random.choice(negative_transform_options, size=num_augmentations, replace=False)
        )

        unaltered_image = batch[i]
        altered_image = random_transforms(unaltered_image)
        negative_pairs.append((unaltered_image, altered_image))
    
    return negative_pairs


def prepare_input(pairs, num_positive_pairs):
    # Concatenating image pairs
    input_data = torch.cat([torch.cat([img1, img2], dim=0) for img1, img2 in pairs], dim=0)

    # Creating target labels based on the number of positive and negative pairs
    num_negative_pairs = len(pairs) - num_positive_pairs
    target_labels = torch.tensor([1] * num_positive_pairs + [0] * num_negative_pairs, dtype=torch.float32)

    return input_data, target_labels

def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2, dim=1)

def get_top_k_indices(query_features, all_features, k):
    # Compute similarity scores between query features and all features
    similarities = cosine_similarity(query_features.unsqueeze(0), all_features)
    # Find top-k indices based on similarity scores
    _, top_k_indices = torch.topk(similarities, k, largest=True)
    return top_k_indices

def retrieve_images(dataset, indices):
    # Fetch images from the dataset based on indices
    images = [dataset[idx] for idx in indices]
    return images

def save_query_and_similar_images(query_image, similar_images, save_path, file_name):
    # TODO: this function assumes that 
    # query_image and similar_images are PIL images or tensors that can be converted to PIL images
    fig, axs = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))
    axs[0].imshow(query_image.permute(1, 2, 0))  # Convert tensor to image
    axs[0].set_title("Query Image")
    axs[0].axis('off')

    for i, img in enumerate(similar_images):
        axs[i + 1].imshow(img.permute(1, 2, 0))  # Convert tensor to image
        axs[i + 1].set_title(f"Similar {i+1}")
        axs[i + 1].axis('off')

    plt.savefig(f"{save_path}/{file_name}")
    plt.close()

def test(model, test_loader, all_features, k=5, validation=False, save_path='saved_images'):
    # test should likely run inference with query image, and save an image in a designated
    # directory that has the query image and the top-k similar images

    # if validation == True, the function is getting called during training
    # TODO: evaluate the effectiveness of yielding accuracy on augmented pos/neg labels
    #       if it is unuseful, this function will probably have to either pick a different
    #       performance metric or just return the loss values
    model.eval()
    total_accuracy = 0
    total_samples = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, (query_images, labels) in enumerate(test_loader):
            query_features = model(query_images)

            for j, query_feature in enumerate(query_features):
                if validation:
                    # Calculate accuracy for augmented labels
                    top_k_indices = get_top_k_indices(query_feature, all_features, k)
                    correct = labels[j] in top_k_indices
                    total_accuracy += correct.item()
                    total_samples += 1
                else:
                    # Find and save top-k similar images
                    top_k_indices = get_top_k_indices(query_feature, all_features, k)
                    similar_images = [test_loader.dataset[idx][0] for idx in top_k_indices]  # Retrieve images based on indices
                    save_query_and_similar_images(query_images[j], similar_images, save_path, f'query_{i}_{j}.png')

    if validation:
        return total_accuracy / total_samples
    else:
        return None
    
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_loss']

def save_best_model(model, optimizer, epoch, best_loss, filename='best_model.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def early_stopping(current_loss, best_loss, threshold=0.01):
    # Stop training if the loss improvement is less than the threshold
    return current_loss < best_loss - threshold

# this function is the loose placeholder logic
def train(model, train_loader, val_loader, optim, criterion, start_epoch=0, num_epochs=10, 
          num_augmentations=3, validate_interval=5, best_loss=np.Inf, checkpoint_file=None):
    
    if checkpoint_file:
        start_epoch, best_loss = load_checkpoint(model, optim, checkpoint_file)

    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()
            
            # TODO: to cut down on training times we may want to set one set of indices
            # to have a positive pair, and the others to have a negative pair (ie pass the indices
            # to the respective pair generating function), would change lines 69/43

            # for each image in the batch we generate a set of pairs of images
            positive_pairs = generate_positive_pairs(batch, num_augmentations)
            negative_pairs = generate_negative_pairs(batch, num_augmentations)

            training_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
            np.random.shuffle(training_pairs)
            input_data, target_labels = prepare_input(training_pairs, len(positive_pairs))
            
            output = model(input_data)
            loss = criterion(output, target_labels)
            loss.backward()
            optim.step()

        if ((epoch % validate_interval) == 0):
            current_loss = test(model, val_loader)

            if (current_loss < best_loss):
                best_loss = current_loss
                save_best_model(model, optim, epoch, best_loss)
            
            if early_stopping(current_loss, best_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

