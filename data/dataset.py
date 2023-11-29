import datasets
from datasets import load_dataset
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import subprocess


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

Dependencies: 
- pip install datasets
- pip install kaggle

This file's purpose is to acquire the necessary data if it does not
yet exist on the filesystem.

The expected disk format (for the data-preprocessing) is each respective
platform's default representation.

Datasets used:
https://huggingface.co/datasets/detectors/lsun_c-ood
https://huggingface.co/datasets/detectors/lsun_r-ood
https://www.kaggle.com/datasets/ashwingupta3012/human-faces/
'''


class CBRDataSet(Dataset):
    def __init__(self, data_dir_list, unified_dir, transforms):
        self.data_dir_list = data_dir_list
        self.unified_dir = unified_dir
        self.transforms = transforms
        
        # mainly for if we scrape the web
        self.valid_file_types = ['.jpg', '.jpeg', '.png']
        self.download()

        if (not os.path.exists(self.unified_dir)):
            os.mkdir(self.unified_dir)

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

    def download(self):
        lsun1_path = "./datasets/lsun1"
        lsun2_path = "./datasets/lsun2"
        faces_path = "./datasets/faces"

        if ((not os.path.exists(lsun1_path)) or (len(os.listdir(lsun1_path)) == 0)):
            lsun1_data = load_dataset("detectors/lsun_c-ood")
            lsun1_data.save_to_disk(lsun1_path)
        else:
            print("Skipping lsun1 download")
        
        if ((not os.path.exists(lsun2_path)) or (len(os.listdir(lsun2_path)) == 0)):
            lsun2_data = load_dataset("detectors/lsun_r-ood")
            lsun2_data.save_to_disk(lsun2_path)
        else:
            print("Skipping lsun2 download")

        if ((not os.path.exists(faces_path)) or (len(os.listdir(faces_path)) == 0)):
            os.mkdir(faces_path)
            download_cmd = 'kaggle datasets download -d ashwingupta3012/human-faces -p %s' % faces_path
            unzip_cmd = 'unzip %s/human-faces.zip -d %s' % (faces_path, faces_path)
            result = subprocess.run(download_cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            result = subprocess.run(unzip_cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        else:
            print("Skipping face data download")
        print("Datasets are up to date!")

    # extract, reshape, and combine the images across multiple datasets
    def preprocess_datasets(self):
        img_index = 0
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
                    img = img.resize((36,36))
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
                    img = img.resize((36,36))
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
if __name__=='__main__':
    ls = [('./datasets/lsun1/train', 'hf'),
          ('./datasets/lsun2/train', 'hf'),
          ('./datasets/faces/Humans', 'kaggle')]
    dir = './datasets/data'
    test = CBRDataSet(ls, dir, transforms=transforms.ToTensor())
    dataloader = DataLoader(test, batch_size=4)
    for x in iter(dataloader):
        print(x.shape)