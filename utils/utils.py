import torch.nn as nn
import torch
import glob
import cv2
import numpy as np
from skimage import io
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
import torch.nn.functional as F

import os
from typing import List


def muti_bce_loss_fusion(d0: float, d1: float, d2: float, d3: float, d4: float, d5: float, d6: float, labels_v: object):
    """
        Method to compute the loss function for the U2Net model
    """
    bce_loss = nn.BCELoss(size_average=True)

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(),
    loss6.data.item()))

    return loss0, loss


def evaluate(true_label_dir: str, pred_label_dir: str) -> [float, float]:
    """
        Method to evaluate the results of the segmentation
    """
    true_img_list = glob.glob(true_label_dir + '/*.png')
    error = []
    fscore = []
    for img in true_img_list:
        pred_img = cv2.imread(os.path.join(pred_label_dir, os.path.basename(img)), cv2.IMREAD_GRAYSCALE)
        pred_img_bin = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)[1]/255
        true_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        true_img_bin = cv2.threshold(true_img, 127, 255, cv2.THRESH_BINARY)[1]/255
        error.append(mean_absolute_error(1-true_img_bin, pred_img_bin))
        fscore.append(f1_score(1-true_img_bin, pred_img_bin, average='micro'))
    return error, fscore


def normPRED(d: List) -> List:
    """
            Method to normalise the predictions
    """
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name: str, pred: object, d_dir: str):
    """
        Method to save the output binary mask as a png file
    """
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    ret, im = cv2.threshold(predict_np * 255, 127, 255, cv2.THRESH_BINARY)

    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)

    imo = cv2.resize(im, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    blur = ((1, 1), 1)
    erode_ = (5, 5)
    dilate_ = (7, 7)
    cv2.imwrite(d_dir + '/' + imidx + '.png', cv2.dilate(
        cv2.erode(
            cv2.GaussianBlur(imo, blur[0], blur[1]),
            np.ones(erode_)),
        np.ones(dilate_)) * 255)


def dice_loss(pred, target, smooth=1.):
    """
            Method to compute the dice loss used in U2Net
    """

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
            Method to compute the loss used in FCN
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
