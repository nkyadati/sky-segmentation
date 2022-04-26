import glob
from torchvision import transforms
from dataset import SegDataset
from torch.utils.data import DataLoader
from utils.transformations import RescaleT, ToTensorLab, RandomCrop

from typing import List, Tuple


class DataLoad:
    """
    Dataloader class with methods to read and load the dataset
    """

    def __init__(self, config: object, flag: str):
        """
        Constructor for the data loader class
        :param config: Object with parameters related to the dataset
        """
        self.config = config
        self.flag = flag

    def load_data(self) -> Tuple[int, List, DataLoader]:
        """
        Method to load the dataset and create a PyTorch data loader
        :return: Tuple with the following - num_samples (Number of samples), img_name_list (List of images), dataloader (PyTorch data loader object)
        """
        image_ext = self.config.image_ext
        label_ext = self.config.label_ext
        data_dir = self.config.data_dir
        if self.flag == "train":
            image_dir = self.config.train_image_dir
            label_dir = self.config.train_label_dir
            batch_size = self.config.train_batch_size
        elif self.flag == "val":
            image_dir = self.config.val_image_dir
            label_dir = self.config.val_label_dir
            batch_size = self.config.val_batch_size

        img_name_list = glob.glob(data_dir + image_dir + '*' + image_ext)
        lbl_name_list = glob.glob(data_dir + label_dir + '*' + label_ext)

        print("---")
        print("Number of images: ", len(img_name_list))
        print("Number of labels: ", len(lbl_name_list))
        print("---")

        num_samples = len(img_name_list)

        if self.flag == "val":
            data = SegDataset(img_name_list=img_name_list, lbl_name_list=lbl_name_list,
                                        transform=transforms.Compose([RescaleT(512), ToTensorLab(flag=0)]))
            dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        else:
            data = SegDataset(
                img_name_list=img_name_list,
                lbl_name_list=lbl_name_list,
                transform=transforms.Compose([
                    RescaleT(320),
                    RandomCrop(288),
                    ToTensorLab(flag=0)]))
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

        return num_samples, img_name_list, dataloader
