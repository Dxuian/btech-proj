#! /usr/bin/python3.6


"""
Descriptions
============================================================
#  Create time  : Jan 15 2021
#  Modified time: Jan 15 2021
============================================================

"""

# This is the DEMO code for the PV module of BILST model.

import logging
import torch
import torch.nn as nn

from torch.autograd import Variable
from collections import OrderedDict
from pathlib import Path
from torch.nn import functional as tf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_hidden(x, hidden_size: int):

    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

    def forward(self, input_data):

        input_weighted = Variable(torch.zeros(input_data.size(0), self.T, self.input_size))
        input_encoded  = Variable(torch.zeros(input_data.size(0), self.T, self.hidden_size))

        hidden = init_hidden(input_data, self.hidden_size).to(device)
        cell   = init_hidden(input_data, self.hidden_size).to(device) 

        for t in range(self.T):

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2)  

            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))  
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]

            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats = 1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 8)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):

        hidden = init_hidden(input_encoded, self.decoder_hidden_size).to(device) 
        cell = init_hidden(input_encoded, self.decoder_hidden_size).to(device) 
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))
        input_encoded = input_encoded.to(device)
        for t in range(self.T):

            x = torch.cat((hidden.repeat(self.T , 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)

            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T),
                    dim=1)  

            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)
            y_history_step = y_history[:, t].unsqueeze(1)

            y_tilde = self.fc(torch.cat((context,y_history_step), dim = 1))  # (batch_size, out_size)

            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0] 
            cell   = lstm_output[1]  


        final_output = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return final_output
