#! /usr/bin/python3.6


"""
Descriptions
============================================================
#  Create time  : Jan 15 2021
#  Modified time: Jan 15 2021
============================================================

"""
# This is a DEMO code for showing the sky image module of the BILST model.


import logging
from   pathlib import Path
from   torch.autograd import Variable
from   collections import OrderedDict

import torch.nn.functional as F
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):

        x = self.relu(x)
        x = self.conv1(x)

        return x

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual
        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x


class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad = True)  # define the trainable parameter

    def forward(self, x):

        x = x * self.weights 

        return x
class st_pv(nn.Module):
    def __init__(self, batch_num):
        super(st_pv,self).__init__()


class ModelImg(nn.Module):
    def __init__(self, c_conf = (3, 2, 32, 32)):

        super(ModelImg, self).__init__()
        logger = logging.getLogger(__name__)
        logger.info('initializing net params and ops ...')

        self.c_conf = c_conf


        self.nb_flow, self.map_height, self.map_width = c_conf[1], c_conf[2], c_conf[3]

        self.relu      = torch.relu
        self.tanh      = torch.tanh
        self.sigmoid   = torch.sigmoid

        self.input_dim   = 1
        self.hidden_dim  = 64

        self.lstm_conv1  = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size = (1, 14), padding = 0)
        self.lstm_conv2  = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size = (1, 14), padding = 0)
        self.lstm_conv3  = nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size = (1, 14), padding = 0)

        self.lstm        = nn.LSTM(4 , self.hidden_dim, batch_first=True)
        self.hidden2out  = nn.Linear(self.hidden_dim, 1)
        self.hidden      = self.init_hidden()
        self.batchnorm   = nn.BatchNorm1d(32)
        self.conv2       = nn.Conv2d(1, 32, kernel_size = (16, 3), padding = (0, 1))
        self.fc1         = nn.Linear(3 * 64 * 64, 120)
        self.fc2         = nn.Linear(120, 32)
        self.fc3         = nn.Linear(32, 3)
        self.fc4         = nn.Linear(12288, 1)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = ( 2* self.c_conf[0]-1) * self.nb_flow)


        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias = True)),
                ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias = True)),
                ('relu2', nn.ReLU()),
            ]))

    def init_hidden(self):

        return Variable(torch.zeros(1, 1, self.hidden_dim , device = device))


    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels = in_channels, out_channels = 64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels = 64, out_channels = self.nb_flow)),
            ('FusionLayer', TrainableEltwiseLayer(n = self.nb_flow, h = self.map_height, w = self.map_width))
        ]))


    def forward(self, input_c):

        main_output = 0

        input_c = input_c.view(-1, (2* self.c_conf[0] - 1) *self.nb_flow, self.map_height, self.map_width)
        out_c = self.c_way(input_c)
        main_output += out_c


        main_output = self.relu(main_output)

        return main_output

