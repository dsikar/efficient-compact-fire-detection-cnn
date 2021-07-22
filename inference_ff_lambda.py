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
import boto3
# use the AWS Lambda-friendly idiom
import PIL.Image as Image

##########################################################################

import torch
import torchvision.transforms as transforms
import shufflenetv2
# start with shufflenetv2 only - S3 has been caching models so let's avoid
# headaches for now
# from models import nasnet_mobile_onfire

##########################################################################


s3 = boto3.resource('s3')
client_s3 = boto3.client('s3')
result = client_s3.download_file("dsikar.models.bucket",'shufflenetv2.py', "/models/shufflenetv2.py")
result = client_s3.download_file("dsikar.models.bucket",'shufflenet_ff.pt', "/weights/shufflenet_ff.pt")

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

def run_model_img(frame, model):
    output = model(frame)
    pred = torch.round(torch.sigmoid(output))
    return pred


##########################################################################

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    img = readImageFromBucket(key, bucket_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # model load
    model = shufflenetv2.shufflenet_v2_x0_5(
        pretrained=False, layers=[
            4, 8, 4], output_channels=[
            24, 48, 96, 192, 64], num_classes=1);

    w_path = '/weights/shufflenet_ff.pt'
    model.load_state_dict(torch.load(w_path, map_location=device));

    np_transforms = data_transform_shufflenetonfire(model);

    #print(f'|__Model loading: {model}')
    model.eval();
    model.to(device);

    #print('\t|____Image processing: ', imgpath)
    #    start_t = time.time()
        # im is a path to image
    # frame = cv2.imread(imgpath)
    frame = np.array(img)
    small_frame = read_img(frame, np_transforms)
    prediction = run_model_img(small_frame, model)
    canonical_prediction = 'NO FIRE'
    if prediction == 0:
        canonical_prediction = 'FIRE'
    #print(canonical_prediction)
    #print("prediction: ", prediction)
    print('ImageName: {0}, Prediction: {1}'.format(key, canonical_prediction))
    response_str = 'ImageName: {0}, Prediction: {1}'.format(key, canonical_prediction)
    filename = key + '.txt'
    client_s3.put_object(Body=response_str, Bucket='tensorflow-images-predictions-dsikar', Key=filename)

def readImageFromBucket(key, bucket_name):
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(key)
    response = object.get()
    return Image.open(response['Body'])

