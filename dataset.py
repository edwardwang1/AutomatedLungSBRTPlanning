from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

class VolumesFromList(Dataset):
    def __init__(self, dataDirectory, list_path):
        self.dataDirectory = dataDirectory

        with open(list_path) as f:
            self.lines = f.read().splitlines()

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.lines)

    def __getitem__(self, idx):
        volumes = np.load(os.path.join(self.dataDirectory, self.lines[idx] + ".npy"))
        return volumes

class VolumesFromListHDUnet(Dataset):
    def __init__(self, dataDirectoryCombined, dataDirectorySeparate, list_path):
        self.dataDirectory = dataDirectoryCombined
        self.dataDirectorySeparate = dataDirectorySeparate

        with open(list_path) as f:
            self.lines = f.read().splitlines()

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.lines)

    def __getitem__(self, idx):
        volumes_combined = np.load(os.path.join(self.dataDirectory, self.lines[idx] + ".npy"))
        volumes_separate = np.load(os.path.join(self.dataDirectorySeparate, self.lines[idx] + ".npy"))
        return np.concatenate((volumes_combined, volumes_separate), axis=0)

class VolumesAndAnglesFromList(Dataset):
    def __init__(self, dataDirectory, angle_path, list_path):
        self.dataDirectory = dataDirectory

        self.df = pd.read_csv(angle_path)

        with open(list_path) as f:
            self.lines = f.read().splitlines()

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.lines)

    def __getitem__(self, idx):
        volumes = np.load(os.path.join(self.dataDirectory, self.lines[idx] + ".npy"))

        min_angle = self.df.loc[self.df['ID'] == self.lines[idx]]["MinAngle"].values[0]
        angular_range = self.df.loc[self.df['ID'] == self.lines[idx]]["Range"].values[0]

        return volumes, min_angle/360, angular_range/360