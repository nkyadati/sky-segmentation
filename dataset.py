from torch.utils.data import Dataset
from skimage import io
import numpy as np
from typing import List, Dict


class SegDataset(Dataset):
    """
        Class for the dataset
    """
    def __init__(self, img_name_list: List, lbl_name_list: List, transform: object = None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self) -> int:
        """
            Function to return the length of a given list
        """
        return len(self.image_name_list)

    def __getitem__(self, idx: int) -> Dict:
        """
            Method to return specific item of a list as a dictionary
        """
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
