import glob
import os
from torchvision import transforms
from dataset import SegDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from utils.transformations import RescaleT, ToTensorLab
from models.fcn import FCN
from models.u2net import U2NET
from models.unet import UNet
from utils.utils import normPRED, save_output, evaluate
import numpy as np


def predict(model_type, img_list, dataloader, predictions_dir):
    """
    Method to make predictions and evaluate the model on a test set

    """
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    model_name = './saved_models/' + model_type + ".pth"
    if not os.path.exists(model_name):
        print("Please place the trained model in ./saved_models and then run the evaluation")
        exit(0)

    if model_type == 'u2net':
        model = U2NET(3, 1)
    elif model_type == 'unet':
        model = UNet(1)
    elif model_type == 'fcn':
        model = FCN(1)
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    for i_test, data_test in enumerate(dataloader):

        print("inferencing:", img_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        if model_type == 'u2net':
            out, d2, d3, d4, d5, d6, d7 = model(inputs_test)
        else:
            out = model(inputs_test)

        pred = 1.0 - out[:, 0, :, :]
        pred = normPRED(pred)

        save_output(img_list[i_test], pred, predictions_dir)


def load_data(image_dir, label_dir, image_ext, label_ext):
    """
    Method to load the dataset and create a PyTorch data loader

    """

    img_name_list = glob.glob(image_dir + '*' + image_ext)

    lbl_name_list = glob.glob(label_dir + '*' + label_ext)

    print("---")
    print("Number of images: ", len(img_name_list))
    print("---")

    num_samples = len(img_name_list)

    data = SegDataset(img_name_list=img_name_list, lbl_name_list=lbl_name_list,
                      transform=transforms.Compose([RescaleT(512), ToTensorLab(flag=0)]))
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    return num_samples, img_name_list, dataloader


image_dir = './datasets/ade20k/val/original/'  # Path to the folder containing the images you want to test on
image_ext = '.jpg'  # File extension of the images in your dataset
label_dir = './datasets/ade20k/val/mask/'  # Path to the ground-truth masks of your dataset. Give the same path as above (image_dir) if you do not want to compute the evaluation metrics
label_ext = '.png'  # File extension of the of the ground-truth masks. Give the same extension as above (image_ext) if you do not have the labels
pred_dir = './results/'  # Path to the folder where you want to store the predictions.
model_name = 'u2net'  # Type of model: u2net or unet or fcn

num_samples, img_name_list, dataloader = load_data(image_dir, label_dir, image_ext, label_ext)
predict(model_name, img_name_list, dataloader, pred_dir)

if image_dir == label_dir:
    exit(0)
else:
    error, fscore = evaluate(label_dir, pred_dir)
    print("MAE (mean): {}; MAE (std): {}".format(np.mean(error), np.std(error)))
    print("FScore (mean): {}; FScore (std): {}".format(np.mean(fscore), np.std(fscore)))
