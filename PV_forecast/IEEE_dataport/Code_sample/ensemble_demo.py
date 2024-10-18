#! /usr/bin/python3.6


"""
Descriptions
============================================================
#  Create time  : Jan 15 2021
#  Modified time: Jan 15 2021
============================================================

"""

!# This is a DEMO code for showing the ensemble module of the BILST model.

import  logging
import  torch
import  torch.nn as nn
import  torchvision
from    torch.autograd import Variable
from    collections import OrderedDict
from    pathlib import Path
from    torch.nn import functional as tf
import  utils
from    image_demo import ModelImg
import  platform
from    config import DefaultConfig


logger    = utils.setup_log()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size       = 32
training_length  = 10
forecast_horizon = 1

num_epoch        = 20
len_closeness    = training_length  
len_period       = int(training_length / 2 )+1
len_trend        = int(training_length / 2 )+1
nb_residual_unit      = 4  
map_height, map_width = 64, 64  # grid size
nb_flow       = 1
learning_rate = 0.0001
trend_ratio   = 3



class MyEnsemble(nn.Module):

    def __init__(self, model_pv_encoder, model_pv_decoder):
        
        super(MyEnsemble, self).__init__()
        self.model_pv_encoder = model_pv_encoder
        self.model_pv_decoder = model_pv_decoder
        self.model_img = model_img
        self.pool      = nn.MaxPool2d(2, 2)
        self.conv1     = nn.Conv2d(1, 8, 8, stride = 4)
        self.bn1       = nn.BatchNorm2d(8)
        self.bn2       = nn.BatchNorm2d(16)
        self.conv2     = nn.Conv2d(8, 16, 4)
        self.fc1       = nn.Linear(8 + 600, 32)
        self.fc2       = nn.Linear(32, 1)


    def forward(self, pv, gatton_pv, img, exo_info):

        img_output   = model_img(img) 
        gatton = torch.reshape(gatton_pv, (1, 60, 10, 1))
        gatton = torch.mean(gatton, dim = 1)
        output = torch.cat((gatton, pv, exo_info), dim = 2)
        gatton_pv = gatton_pv.squeeze(2)
        _, input_encoded = self.model_pv_encoder(output)
        decoder_output   = self.model_pv_decoder(input_encoded, pv[:, :, -1])
        cat_output = torch.cat((gatton_pv, img_output, decoder_output), dim = 1)
        a = tf.relu(self.fc1(cat_output))
        b = self.fc2(a)
        final_output = b.squeeze(1)

        return final_output
