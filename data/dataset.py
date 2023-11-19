import datasets
import glob
import itertools
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


'''
This file defines our custom pytorch Dataset class that conglomerates
and accesses the files downloaded across many datasets. As conten-based
retrieval can be structured as an unsupervised or self-supervised task,
we do not need ot create a mapping between images and their respective
ground truth labels.

Current conglomeration approach:
Huggingface stores their datasets on the filesystem as a .hf face, so for
each HF dataset given, we extract and resave each datapoint as its own file
to avoid needing to load the dataset again on __getitem__. For simplicitly,
we move each non HF dataset to the same directory as well.

For now, all images are saved as 32x32 RGB to match that of the huggingface
datasets.
'''


class CBRDataSet(Dataset):
    def __init__(self, data_dir_list, unified_dir, transforms):
        self.data_dir_list = data_dir_list
        self.unified_dir = unified_dir
        self.transforms = transforms
        
        # mainly for if we scrape the web
        self.valid_file_types = ['.jpg', '.jpeg', '.png']
        
        if (len(os.listdir(self.unified_dir)) == 0):
            print("Pre-processing data!")
            self.preprocess_datasets()
        
        self.image_files = self.get_images()
        self.num_images = len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        if self.transforms:
            img = self.transforms(img)
        return img

    def __len__(self):
        return self.num_images 

    # extract, reshape, and combine the images across multiple datasets
    def preprocess_datasets(self):
        img_index = 0
        if (not os.path.exists(self.unified_dir)):
            os.mkdir(self.unified_dir)

        for (dir, type) in self.data_dir_list:
            if (type == 'hf'):
                dataset = datasets.load_from_disk(dir)
                for img_dict in dataset:
                    # extract file name information and build new filename
                    img_name = "%d.jpg" % img_index
                    img_file = os.path.join(self.unified_dir, img_name)
                    img = img_dict['image']
                
                    # reshape our image/convert to RGB and save
                    img = img.convert('RGB')
                    img.save(img_file)
                    img_index += 1
            elif (type == 'kaggle'):
                imgs = os.listdir(dir)
                for img_file in imgs:
                    # extract file name information and build new filename
                    file, type = os.path.splitext(img_file)
                    name = os.path.join(dir, img_file)
                    img = Image.open(name)
                    new_img = str(img_index) + type
                    new_img_file = os.path.join(self.unified_dir, new_img)
                    
                    # reshape our image/convert to RGB and save
                    img = img.convert('RGB')
                    img = img.resize((32,32))
                    img.save(new_img_file)
                    img_index += 1

    # walk through each file names in the pre-processed dir and collect valid files
    def get_images(self):
        images = []
        files = os.listdir(self.unified_dir)
        for img_file in files:
            file, type = os.path.splitext(img_file)
            if (img_file == ".DS_Store") or (type not in self.valid_file_types):
                continue
            file_name = os.path.join(self.unified_dir, img_file)
            images.append(file_name)
        return images


# simple collation function to be used in the future for the DataLoader
# (I believe this is the same as the default collate_fn)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    return images, labels


# example build
# if __name__=='__main__':
#     ls = [('./datasets/lsun1/train', 'hf'),
#           ('./datasets/lsun2/train', 'hf'),
#           ('./datasets/faces/Humans', 'kaggle')]
#     dir = './datasets/data'
#     test = CBRDataSet(ls, dir, None)