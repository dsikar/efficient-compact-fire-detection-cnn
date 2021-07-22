##########################################################################

# Example : perform live fire detection in image/video/webcam using
# NasNet-A-OnFire, ShuffleNetV2-OnFire CNN models.

# Copyright (c) 2020/21 - William Thompson / Neelanjan Bhowmik / Toby
# Breckon, Durham University, UK

# License :
# https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn/blob/main/LICENSE

##########################################################################

import cv2
import os
import sys
import math
# from PIL import Image
import argparse
import time
import numpy as np
import math
# for our lambda function
import json
# import boto3
# use the AWS Lambda-friendly idiom
import PIL.Image as Image

##########################################################################

import torch
import torchvision.transforms as transforms
from models import shufflenetv2
# start with shufflenetv2 only - S3 has been caching models so let's avoid
# headaches for now
# from models import nasnet_mobile_onfire

##########################################################################


def data_transform(model):
    # transforms needed for shufflenetonfire
    if model == 'shufflenetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    # transforms needed for nasnetonfire
    if model == 'nasnetonfire':
        np_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return np_transforms

##########################################################################

def data_transform_shufflenetonfire(model):
    # transforms needed for shufflenetonfire
    np_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return np_transforms
    
##########################################################################

# read/process image and apply tranformation


def read_img(frame, np_transforms):
    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    small_frame = Image.fromarray(small_frame)
    small_frame = np_transforms(small_frame).float()
    small_frame = small_frame.unsqueeze(0)
    small_frame = small_frame.to(device)

    return small_frame

##########################################################################

# model prediction on image


def run_model_img(args, frame, model):
    output = model(frame)
    pred = torch.round(torch.sigmoid(output))
    return pred


##########################################################################
# parse command line arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        help="Path to image file or image directory")

    args = parser.parse_args()
    #print(f'\n{args}')
    imgpath = args.image

##########################################################################

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #print('\n\nBegin {fire, no-fire} classification :')

    # model load
    model = shufflenetv2.shufflenet_v2_x0_5(
        pretrained=False, layers=[
            4, 8, 4], output_channels=[
            24, 48, 96, 192, 64], num_classes=1);

    w_path = './weights/shufflenet_ff.pt'
    model.load_state_dict(torch.load(w_path, map_location=device));

    np_transforms = data_transform_shufflenetonfire(model);

    #print(f'|__Model loading: {model}')
    model.eval();
    model.to(device);

    #print('\t|____Image processing: ', imgpath)
    #    start_t = time.time()
        # im is a path to image
    frame = cv2.imread(imgpath)
    small_frame = read_img(frame, np_transforms)
    prediction = run_model_img(args, small_frame, model)
    canonical_prediction = 'NO FIRE'
    if prediction == 0:
        canonical_prediction = 'FIRE'
    print(canonical_prediction)
    #print("prediction: ", prediction)
        

