from datasets import load_dataset
import os
import subprocess

'''
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


def main():
    lsun1_path = "./datasets/lsun1"
    lsun2_path = "./datasets/lsun2"
    faces_path = "./datasets/faces"

    if ((not os.path.exists(lsun1_path)) or (len(os.listdir(lsun1_path)) == 0)):
        lsun1_data = load_dataset("detectors/lsun_c-ood")
        lsun1_data.save_to_disk(lsun1_path)
    if ((not os.path.exists(lsun2_path)) or (len(os.listdir(lsun2_path)) == 0)):
        lsun2_data = load_dataset("detectors/lsun_r-ood")
        lsun2_data.save_to_disk(lsun2_path)
    if ((not os.path.exists(faces_path)) or (len(os.listdir(faces_path)) == 0)):
        os.mkdir(faces_path)
        download_cmd = 'kaggle datasets download -d ashwingupta3012/human-faces -p %s' % faces_path
        unzip_cmd = 'unzip %s/human-faces.zip -d %s' % (faces_path, faces_path)
        result = subprocess.run(download_cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        result = subprocess.run(unzip_cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
    print("Datasets are up to date!")

if __name__=='__main__':
    main()