import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import imageio
import torch

class LSUNDataset(Dataset):

    def __init__(self, filename, data_path, transform):
        self.files = np.loadtxt(filename, dtype=str)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_full = f'{self.data_path}/{file}'

        image = Image.fromarray(imageio.imread(file_full))

        if self.transform:
            return [self.transform(image)]
        
        return [image]