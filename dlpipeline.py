from typing import List
from dataloader import DataLoad
from dlmodel import DLModel


class DLPipeline:
    """
    Class running the different steps of the deep learning pipeline
    """

    def __init__(self, loader: DataLoad, dlmodel: DLModel):
        """
        Constructor for the DLPipeline class
        :param loader: DataLoader object
        :param dlmodel: DLModel object
        """
        self.loader = loader
        self.dlmodel = dlmodel

    def run_training(self, n_channels: int, n_classes: int) -> None:
        """
        Method to run the steps of the deep learning pipeline - load data, extract features, train and evaluate the model
        :param n_classes: Number of output classes
        :param n_channels: Number of channels in the input dataset
        :return:
        """
        print("Running DL pipeline")
        # Load the features if the file exists, else call the feature extraction function
        num_samples, img_name_list, dataloader = self._load_data()
        self.dlmodel.build(n_channels, n_classes)
        self._train(dataloader)

    def run_predict_eval(self, n_channels: int, n_classes: int) -> None:
        """
        Method to run the steps of the deep learning pipeline - load data, extract features, train and evaluate the model
        :param n_classes: Number of output classes
        :param n_channels: Number of channels in the input dataset
        :return:
        """
        print("Running DL pipeline")
        # Load the features if the file exists, else call the feature extraction function
        num_samples, img_name_list, dataloader = self._load_data()
        self.dlmodel.build(n_channels, n_classes)
        self._evaluate(dataloader, num_samples, img_name_list)

    def _load_data(self) -> dict:
        """
        Private method to call the load_data method of the DataLoader class
        :return:
        """
        return self.loader.load_data()

    def _train(self, dataloader: dict) -> None:
        """
        Private method calling the build, train, and evaluate methods of the DLModel class
        :param dataloader: PyTorch data loader object
        """
        self.dlmodel.train(len(dataloader.dataset), dataloader)

    def _evaluate(self, dataloader: dict, num_samples: int, img_name_list: List) -> None:
        """
        Private method calling the build, train, and evaluate methods of the DLModel class
        :param dataloader: PyTorch data loader object
        :param num_samples: Number of samples in the test dataset
        :param img_name_list: List of images in the test dataset
        """
        self.dlmodel.predict(num_samples, img_name_list, dataloader)
