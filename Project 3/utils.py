import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    def __init__(
        self,
        csv_file="TrainingDataSet.csv",
        root_dir_noisy="TrainingDataSet",
        root_dir_ref="./",
        transform=None,
    ):
        # read csv file
        self.name_csv = pd.read_csv(csv_file)

        # store attributes
        self.root_dir_noisy = root_dir_noisy
        self.root_dir_ref = root_dir_ref
        self.transform = transform

    def __len__(self):
        return len(self.name_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image filenames
        ref_img_name = os.path.join(self.root_dir_ref, self.name_csv.iloc[idx, 0])
        noisy_img_name = os.path.join(self.root_dir_noisy, self.name_csv.iloc[idx, 2])

        # load images
        ref_image = read_image(ref_img_name)
        noisy_image = read_image(noisy_img_name)

        # apply transforms
        if self.transform:
            ref_image = self.transform(ref_image)
            noisy_image = self.transform(noisy_image)
        return noisy_image, ref_image
