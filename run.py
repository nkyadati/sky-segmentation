import json
from config import CFG
from utils.utils import evaluate
from dataloader import DataLoad
from dlmodel import U2NetModel, UNetModel, FCNModel
from dlpipeline import DLPipeline

import os
import random
import glob
import numpy as np


class Config:
    """Config class which contains data model parameters"""

    def __init__(self, data: object, model: object):
        """
        Constructor for the configuration file containing all the parameter choices
        :param data: Parameters related to the dataset
        :param model: Hyperparameters for the model
        """
        self.data = data
        self.model = model

    @classmethod
    def from_json(cls, cfg: dict) -> object:
        """
        Class method to create the class structure of the different parameters
        :param cfg: Dictionary imported from the config file
        :return: Python object encapsulating all the four parameter objects
        """
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.model)


class HelperObject(object):
    """
    Helper class to convert json into Python object
    """

    def __init__(self, dict_):
        self.__dict__.update(dict_)


if __name__ == "__main__":
    config = Config.from_json(CFG)

    flag = "val"
    if flag == "train":
        # Create objects for DataLoader, Extractor, Preprocessor, and DLModel
        loader = DataLoad(config.data, flag)

        model_type = config.model.model_name
        if model_type == "u2net":
            dlmodel = U2NetModel(config)
        elif model_type == "unet":
            dlmodel = U2NetModel(config)
        elif model_type == "fcn":
            dlmodel = FCNModel(config)

        # Run the pipeline
        dlpipeline = DLPipeline(loader, dlmodel)
        dlpipeline.run_training(n_channels=3, n_classes=1)
    elif flag == "val":
        loader = DataLoad(config.data, flag)
        model_type = config.model.model_name
        if model_type == "u2net":
            dlmodel = U2NetModel(config)
        elif model_type == "unet":
            dlmodel = UNetModel(config)
        elif model_type == "fcn":
            dlmodel = FCNModel(config)

        dlpipeline = DLPipeline(loader, dlmodel)
        dlpipeline.run_predict_eval(n_channels=3, n_classes=1)
        error, fscore = evaluate(os.path.join(config.data.data_dir, config.data.val_label_dir),
                                 config.data.predictions_dir)
        print("MAE (mean): {}; MAE (std): {}".format(np.mean(error), np.std(error)))
        print("FScore (mean): {}; FScore (std): {}".format(np.mean(fscore), np.std(fscore)))
