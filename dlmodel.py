from abc import ABC
from typing import List
from models.u2net import U2NET
from models.unet import UNet
from models.fcn import FCN
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.utils import muti_bce_loss_fusion
import os
from collections import defaultdict

from utils.utils import normPRED, save_output, calc_loss


class DLModel(ABC):
    """
    Abstract class for the Deep learning model used for training
    """

    def __init__(self, config):
        self.config = config

    def build(self, input_shape, n_outputs):
        pass

    def train(self, num_samples, dataloader):
        pass

    def predict(self, num_samples, img_list, dataloader):
        pass


class U2NetModel(DLModel):
    """
    Class implementing U2Net model and extending the DLModel abstract class
    """

    def __init__(self, config: object):
        """
        Constructor for the U2NetModel class
        :param config: Object containing the configuration parameters
        """
        super().__init__(config)
        self.model = None

    def build(self, n_channels: int, n_outputs: int) -> None:
        """
        Method to construct the U2Net model
        :param n_channels: Number of channels in the input dataset (3 channels for RGB images)
        :param n_outputs: Number of classes
        :return:
        """
        self.model = U2NET(n_channels, n_outputs)

    def train(self, num_samples: int, dataloader: dict):
        """
        Method to compile and train the U2Net model
        :param dataloader: PyTorch data loader for the dataset
        :param num_samples: Number of samples in the dataset for training
        """
        if torch.cuda.is_available():
            self.model.cuda()

        print("---define optimizer...")
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

        for epoch in range(0, self.config.model.epochs):
            self.model.train()

            for i, data in enumerate(dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']

                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                optimizer.zero_grad()

                d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

                loss.backward()
                optimizer.step()

                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()

                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                    epoch + 1, epoch, (i + 1) * self.config.data.train_batch_size, num_samples, ite_num, running_loss / ite_num4val,
                    running_tar_loss / ite_num4val))

                if ite_num % self.config.model.save_frequency == 0:
                    torch.save(self.model.state_dict(),
                               self.config.data.model_dir + self.config.model.model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                               ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    self.model.train()  # resume train
                    ite_num4val = 0

    def predict(self, num_samples: int, img_list: List,  dataloader: dict):
        """
        Method to make predictions and evaluate the model on a test set
        :param dataloader: PyTorch data loader for the dataset
        :param img_list: List of images in the test set
        :param num_samples: Number of images in the test set
        """
        if not os.path.exists(self.config.data.predictions_dir):
            os.mkdir(self.config.data.predictions_dir)

        model = self.config.model.model_dir + self.config.model.model_name + ".pth"
        if not os.path.exists(model):
            print("Please place the trained model in ./saved_models and then run the evaluation")
            exit(0)

        self.model.load_state_dict(torch.load(model, map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        for i_test, data_test in enumerate(dataloader):

            print("inferencing:", img_list[i_test].split(os.sep)[-1])
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 = self.model(inputs_test)

            pred = 1.0 - d1[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_list[i_test], pred, self.config.data.predictions_dir)

            del d1, d2, d3, d4, d5, d6, d7


class UNetModel(DLModel):
    """
    Class implementing U2Net model and extending the DLModel abstract class
    """

    def __init__(self, config: object):
        """
        Constructor for the U2NetModel class
        :param config: Object containing the configuration parameters
        """
        super().__init__(config)
        self.model = None

    def build(self, n_channels: int, n_outputs: int) -> None:
        """
        Method to construct the U2Net model
        :param n_channels: Number of channels in the input dataset (3 channels for RGB images)
        :param n_outputs: Number of classes
        :return:
        """
        self.model = UNet(n_outputs)

    def train(self, num_samples: int, dataloader: dict):
        """
        Method to compile and train the U2Net model
        :param dataloader: PyTorch data loader for the dataset
        :param num_samples: Number of samples in the dataset for training
        """
        if torch.cuda.is_available():
            self.model.cuda()

        print("---define optimizer...")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)

        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        ite_num4val = 0
        save_frq = 1000

        for epoch in range(0, self.config.model.epochs):
            self.model.train()
            metrics = defaultdict(float)
            epoch_samples = 0
            for i, data in enumerate(dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']

                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                optimizer.zero_grad()

                outputs = self.model(inputs_v)
                loss = calc_loss(outputs, labels_v, metrics)

                loss.backward()
                optimizer.step()

                epoch_samples += inputs.size(0)
                running_loss += loss.data.item()

                # del temporary outputs and loss
                del loss

                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f " % (
                    epoch + 1, self.config.model.epochs, (i + 1) * self.config.data.train_batch_size, num_samples,
                    ite_num, running_loss / ite_num4val))

                if ite_num % save_frq == 0:
                    torch.save(self.model.state_dict(),
                               self.config.model.model_dir + self.config.model.model_name + "_bce_itr_%d_train_%3f.pth" % (
                               ite_num, running_loss / ite_num4val))
                    running_loss = 0.0
                    self.model.train()
                    ite_num4val = 0

    def predict(self, num_samples: int, img_list: List,  dataloader: dict):
        """
        Method to make predictions and evaluate the model on a test set
        :param dataloader: PyTorch data loader for the dataset
        :param img_list: List of images in the test set
        :param num_samples: Number of images in the test set
        """
        if not os.path.exists(self.config.data.predictions_dir):
            os.mkdir(self.config.data.predictions_dir)

        model_name = self.config.model.model_dir + self.config.model.model_name + ".pth"
        if not os.path.exists(model_name):
            print("Please place the trained model in ./saved_models and then run the evaluation")
            exit(0)

        self.model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        for i_test, data_test in enumerate(dataloader):

            print("inferencing:", img_list[i_test].split(os.sep)[-1])
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            out = self.model(inputs_test)

            pred = 1.0 - out[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_list[i_test], pred, self.config.data.predictions_dir)


class FCNModel(DLModel):
    """
    Class implementing U2Net model and extending the DLModel abstract class
    """

    def __init__(self, config: object):
        """
        Constructor for the U2NetModel class
        :param config: Object containing the configuration parameters
        """
        super().__init__(config)
        self.model = None

    def build(self, n_channels: int, n_outputs: int) -> None:
        """
        Method to construct the U2Net model
        :param n_channels: Number of channels in the input dataset (3 channels for RGB images)
        :param n_outputs: Number of classes
        :return:
        """
        self.model = FCN(n_outputs)

    def train(self, num_samples: int, dataloader: dict):
        """
        Method to compile and train the U2Net model
        :param dataloader: PyTorch data loader for the dataset
        :param num_samples: Number of samples in the dataset for training
        """
        if torch.cuda.is_available():
            self.model.cuda()

        print("---define optimizer...")
        criterion = torch.nn.BCELoss()

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        ite_num4val = 0
        save_frq = 1000  # save the model every 2000 iterations

        for epoch in range(0, self.config.model.epochs):
            self.model.train()
            epoch_samples = 0
            for i, data in enumerate(dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']

                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                optimizer.zero_grad()

                outputs = self.model(inputs_v)
                loss = criterion(outputs, labels_v)

                loss.backward()
                optimizer.step()

                epoch_samples += inputs.size(0)
                running_loss += loss.data.item()

                del loss

                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
                    epoch + 1, self.config.model.epochs, (i + 1) * self.config.data.train_batch_size, num_samples,
                    ite_num, running_loss / ite_num4val))

                if ite_num % save_frq == 0:
                    torch.save(self.model.state_dict(),
                               self.config.model.model_dir + self.config.model.model_name + "_bce_itr_%d_train_%3f_tar.pth" % (
                               ite_num, running_loss / ite_num4val))
                    running_loss = 0.0
                    self.model.train()
                    ite_num4val = 0

    def predict(self, num_samples: int, img_list: List,  dataloader: dict):
        """
        Method to make predictions and evaluate the model on a test set
        :param dataloader: PyTorch data loader for the dataset
        :param img_list: List of images in the test set
        :param num_samples: Number of images in the test set
        """
        if not os.path.exists(self.config.data.predictions_dir):
            os.mkdir(self.config.data.predictions_dir)

        model_name = self.config.model.model_dir + self.config.model.model_name + ".pth"
        if not os.path.exists(model_name):
            print("Please place the trained model in ./saved_models and then run the evaluation")
            exit(0)

        self.model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        for i_test, data_test in enumerate(dataloader):

            print("inferencing:", img_list[i_test].split(os.sep)[-1])
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            out = self.model(inputs_test)

            pred = 1.0 - out[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_list[i_test], pred, self.config.data.predictions_dir)
