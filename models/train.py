import numpy as np
import torch
from torchvision import transforms

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


def prepare_input(pairs):
    # TODO: pairs will be a list of tuples (img1, img2)
    # we want something of the form: input_data = torch.cat([img1, img2], dim=0)

    # for the labels we are can do 1 and 0 based on the number of pairs seen
    # target_labels = torch.tensor([1] * len(positive_pairs) + [0] * len(negative_pairs), dtype=torch.float32)
    pass


def test(model, test_loader, validation=False):
    # test should likely run inference with query image, and save an image in a designated
    # directory that has the query image and the top-k similar images

    # if validation == True, the function is getting called during training
    # TODO: evaluate the effectiveness of yielding accuracy on augmented pos/neg labels
    #       if it is unuseful, this function will probably have to either pick a different
    #       performance metric or just return the loss values
    pass


# this function is the loose placeholder logic
# TODO: tqdm, checkpointing, validation/early stopping
def train(model, train_loader, val_loader, optim, criterion, start_epoch=0, num_epochs=10, 
          num_augmentations=3, validate_interval=5, best_loss=np.Inf):
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            
            # TODO: to cut down on training times we may want to set one set of indices
            # to have a positive pair, and the others to have a negative pair (ie pass the indices
            # to the respective pair generating function), would change lines 69/43

            # for each image in the batch we generate a set of pairs of images
            positive_pairs = generate_positive_pairs(batch, num_augmentations)
            negative_pairs = generate_negative_pairs(batch, num_augmentations)

            training_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
            np.random.shuffle(training_pairs)
            input_data, target_labels = prepare_input(training_pairs)
            
            output = model(input_data)
            loss = criterion(output, target_labels)
            loss.backward()
            optim.step()

        if ((epoch % validate_interval) == 0):
            current_loss = test(model, val_loader)

            if (current_loss < best_loss):
                # TODO: checkpoint model with best_loss and epoch in the dict
            # TODO: check for early stopping
                pass