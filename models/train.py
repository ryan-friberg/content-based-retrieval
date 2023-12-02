import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm 

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


def generate_positive_pairs(batch, indices, num_augmentations):
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
    for idx in indices:
        # randomly sample a set of transforms (to get diverse alterations throughout training)
        random_transforms = transforms.Compose(
            np.random.choice(positive_transform_options, size=num_augmentations, replace=False)
        )

        unaltered_image = batch[idx]
        positive_associated_image = random_transforms(unaltered_image)
        positive_pairs.append((unaltered_image, positive_associated_image))
    
    return positive_pairs


def generate_negative_pairs(batch, labels, indices, num_augmentations, dataset):
    # sample a new random image from the dataset and define a set of transformations 
    # that alter the image a little more drastically to build negative associations

    negative_transform_options = np.array([
        transforms.RandomHorizontalFlip(1.0),
        transforms.RandomVerticalFlip(1.0),
        transforms.RandomRotation(180),
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.1), ratio=(0.3, 3.3), value='random'),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=8, sigma=(0.1, 0.2)),
        AddGaussianNoise(0., 2.0)
    ])

    negative_pairs = []
    for idx in indices:
        unaltered_image = batch[idx]
        image_idx = labels[idx]
    
        # ensure no duplicate (even though the odds are super low)
        found = False
        while not found:
            neg_idx = np.random.randint(0, len(dataset))
            if neg_idx != image_idx:
                negative_associated_image = dataset[neg_idx]

        random_transforms = transforms.Compose(
            np.random.choice(negative_transform_options, size=num_augmentations, replace=False)
        )
        negative_associated_image = random_transforms(unaltered_image)
        
        negative_pairs.append((unaltered_image, negative_associated_image))
    
    return negative_pairs


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


def contrastive_loss(output1, output2, target, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                  (target) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


def test(model, test_loader, test_dataset, num_augmentations, scoring_fn, validation=False):
    # test should likely run inference with query image, and save an image in a designated
    # directory that has the query image and the top-k similar images

    # if validation == True, the function is getting called during training
    # TODO: 
    # - determine if there are any differences for validation runs (ie see if validation bool is necessary)
    # - figure out some performance metric with respect to the pairwaise labels
    #       - one option could be: for any image in the batch, augment it and create a "batch" out of the
    #         remaining images in the batch + the augmented one. The augmented image is the only positive
    #         label. The accuracy can be determine using the scoring_fn passed from main between each looking 
    #         for the closest option

    model.eval()
    total_loss = 0
    for batch, labels in tqdm(enumerate(test_loader), total=len(test_loader)):
        # determine which batch elements of the batch are going to be neg/pos
        batch_indices = torch.randperm(batch.shape[0])
        split_index   = int(batch.shape[0] * 0.5)
        pos_indices   = batch_indices[:split_index]
        neg_indices   = batch_indices[split_index:]

        # for each image in the batch we generate a set of pairs of images
        positive_pairs = generate_positive_pairs(batch, pos_indices, num_augmentations)
        negative_pairs = generate_negative_pairs(batch, labels, neg_indices, num_augmentations, test_dataset)
        pairwise_labels = torch.tensor([1] * len(pos_indices) + [0] * len(neg_indices))
        test_pairs = np.arrray([positive_pairs + negative_pairs])
        np.random.shuffle(zip(test_pairs, pairwise_labels))

        output1 = model(torch.cat([pair[0] for pair in test_pairs], dim=0))
        output2 = model(torch.cat([pair[1] for pair in test_pairs], dim=0))
        loss = contrastive_loss(output1, output2, pairwise_labels, margin=1.0)
        total_loss += loss
    
    return total_loss / len(test_loader)


# this function is the loose placeholder logic
def train(model, train_loader, val_loader, train_dataset, val_dataset, optim, scoring_fn, start_epoch=0, 
          num_epochs=10, num_augmentations=3, validate_interval=5, best_loss=np.Inf):

    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for batch, labels in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()
            
            # determine which batch elements of the batch are going to be neg/pos
            batch_indices = torch.randperm(batch.shape[0])
            split_index   = int(batch.shape[0] * 0.5)
            pos_indices   = batch_indices[:split_index]
            neg_indices   = batch_indices[split_index:]

            # for each image in the batch we generate a set of pairs of images
            positive_pairs = generate_positive_pairs(batch, pos_indices, num_augmentations)
            negative_pairs = generate_negative_pairs(batch, labels, neg_indices, num_augmentations, train_dataset)
            pairwise_labels = torch.tensor([1] * len(pos_indices) + [0] * len(neg_indices))
            train_pairs = np.arrray([positive_pairs + negative_pairs])
            np.random.shuffle(zip(train_pairs, pairwise_labels))

            output1 = model(torch.cat([pair[0] for pair in train_pairs], dim=0))
            output2 = model(torch.cat([pair[1] for pair in train_pairs], dim=0))

            loss = contrastive_loss(output1, output2, pairwise_labels, margin=1.0)
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