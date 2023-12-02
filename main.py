import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data.dataset import GalaxyCBRDataSet, collate_fn
from models.train import train, test
from models.transformer_feature_extractor import FeatureExtractorViT

# define CLI arugments, add more as needed
parser = argparse.ArgumentParser(description='Content-based image retrieval pipeline')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--optim', default='adam', type=str, help='training optimizer: sgd, sgd_nest, adagrad, adadelta, or adam')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--train_split', default=0.8, type=float, help='percentage of dataset to be used for training')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--load', default='', type=str, help='model checkpoint path')
parser.add_argument('--model', default='transformer', type=str, help='select which model architecture')
parser.add_argument('--data_dir', default='./data/galaxy_dataset/', type=str, help='location of data files')
parser.add_argument('--train', action='store_true', type=bool, help='train model')
parser.add_argument('--test', action='store_true', type=bool, help='location of data files')

# currently unused
parser.add_argument('--num_augmentations', default=3, type=int, help='number of augmentations during training')


def build_model(arch_name):
    model = None
    if (arch_name == 'transformer'):
        print("=> Transformer")
        model = FeatureExtractorViT()
    elif (arch_name == 'cnn'):
        print("=> CNN")
        model = None
    return model


def build_optim(optim_name, model, lr):
    # pick the model from the arguments
    optim = None
    if (optim_name == 'sgd'):
        print("=> SGD")
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif (optim_name == 'sgd_nest'):
        print("=> SGD w/ Nesterov")
        optim = torch.optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif (optim_name == 'adam'):
        print("=> Adam")     
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif (optim_name == 'adagrad'):
        print("=> Adagrad")
        optim = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=5e-4)
    elif (optim_name == 'adadelta'):
        print("=> Adadelta")
        optim = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=5e-4)
    return optim


def determine_device(requested_device_name):
    if requested_device_name == 'cuda':
        print("Checking for GPU...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("GPU found, using GPU...")
        else:
            print("GPU not found!")
    else:
        print("Using CPU...")
        device = 'cpu'
    return device


def main():
    args = parser.parse_args()

    # build appropriate model
    print('===> Building model...')
    model = build_model(args.model)
    device = determine_device(args.device)
    model.to(device)

    # load pre-trained checkpoint, if specified
    start = 0
    best_loss = np.Inf
    if args.load != '': # model needs to be loaded to the same device it was saved from
        print('===> Loading checkpoint...')
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model'])
        start = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss'] + 1
        print("=> Loaded!")

    print("===> Building optimizer...")
    optim = build_optim(args.optim, model, args.lr)
    criterion = nn.CrossEntropyLoss() # placeholder for custom contrastive loss fn

    print("===> Building dataset and dataloaders...")
    data_transforms = transforms.ToTensor()
    galaxy_dataset = GalaxyCBRDataSet(args.data_dir, data_transforms)

    train_size = int(args.train_split * len(galaxy_dataset))
    test_val_size = (len(galaxy_dataset) - train_size) / 2
    train_dataset, test_dataset, val_dataset = random_split(galaxy_dataset,[train_size, test_val_size, test_val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    def compute_all_features(model, dataloader, device):
        """TODO: do we need to compute all the features beforehand?"""
        model.eval()
        all_features = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                features = model(images)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        return all_features

    all_features = compute_all_features(model, test_loader, device)

    if (args.train):
        print("===> Training...")
        train(model, test_loader)
    if (args.test):
        print("===> Testing...")
        train(model, train_loader, val_loader, optim, criterion, start_epoch=start, num_epochs=args.epochs, 
              num_augmentations=3, validate_interval=5, best_loss=best_loss)
        accuracy = test(model, test_loader, all_features, k=5, validation=True)
        print(f"Validation Accuracy: {accuracy}")
        print("Training complete!")


if __name__=='__main__':
    main()