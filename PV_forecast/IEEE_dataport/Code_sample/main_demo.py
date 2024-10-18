#! /usr/bin/python3.6


"""
Descriptions
============================================================
#  Create time  : Jan 15 2021
#  Modified time: Jan 15 2021
============================================================

!# This is a DEMO code for showing the main structure of the BILST model.

"""

# Import packages and other modules


import os
import torch
import numpy as np
import math
import torchvision
import pickle
import random
import torch.optim as optim
import torchvision.models as models
import platform
import utils
import torch.nn as nn
import matplotlib.pyplot as plt
from   torch.autograd import Variable
from   torchvision import transforms
from   config import DefaultConfig
from   pv_demo import Encoder, Decoder
from   model_ensemble import MyEnsemble
from   batch_image_generation import batch_image_generation
from   PIL import Image

# Initialization

logger           = utils.setup_log()
imagePath        = "Image path"
logger.info(imagePath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

validation_split    = 0.1
early_stop_patience = 30


def numpy_to_tensor(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':

    logger.info("==========================") 
    logger.info("Bi-level ST Prediction Demo")
    logger.info(f'torch version: %s' %(torch.__version__)) 

    if torch.cuda.is_available():

        logger.info(torch.cuda.get_device_name(0))
        logger.info(torch.cuda.is_available())

    with open('uq_st_pv.pickle', 'rb') as handle:
        uq_st_pv      = pickle.load(handle)    
    with open('gatton_pv.pickle', 'rb') as handle:
        gatton_pv_dict      = pickle.load(handle)    
    with open('exo_train.pickle', 'rb') as handle:
        exo_train      = pickle.load(handle)    

    fileInFolder     = []
    fileSelectedList = []

    for path, subdirs, files in os.walk(imagePath):
        for name in files:
            fileInFolder.append(os.path.join(path, name))

    fileSelectedList = fileInFolder

    # Hyper-parameters

    model_name  = 'bi-level-st'

    logger      = utils.setup_log()
    batch_size       = 32
    training_length  = 10
    forecast_horizon = 15

    num_epoch     = 50
    len_closeness = training_length  
    len_period    = int(training_length / 2 )+1
    len_trend     = int(training_length / 2 )+1
    nb_residual_unit      = 4  
    map_height, map_width = 64, 64  # grid size
    nb_flow       = 1
    learning_rate = 0.0001
    trend_ratio   = 3
    loss_fn       = nn.MSELoss()
    hidden_size   = 256
    img_num       = 6

    encoder_1     = Encoder(input_size, hidden_size, training_length).to(device) 
    decoder_1     = Decoder(hidden_size, hidden_size, training_length).to(device) 
    model         = MyEnsemble(encoder_1, decoder_1).to(device)
    optimizer     = torch.optim.Adam(model.parameters(), lr = learning_rate)


    train_losslist  = []
    valid_loss_list = []

    total_length = len(uq_st_pv)
    
    for epoch in range(num_epoch):

        total_list       = list(range(len(uq_st_pv[0,:])))
        
        trainset_length  = int(0.85 * len(total_list))
        valid_breakpoint = int(0.95 * len(total_list))
        train_index_list = total_list[:trainset_length]
        valid_index_list = total_list[trainset_length: valid_breakpoint]
        
        num_iter         = int(trainset_length / batch_size) - 1
        
        random.shuffle(train_index_list)
        random.shuffle(valid_index_list)

        running_loss_iter  = 0.0
        running_loss_epoch = 0.0

        for k in range(num_iter):
            
            random_num_list   = []
            
            for i in range(batch_size):
                random_num_list.append(train_index_list.pop(0))
                
            model.train()
            optimizer.zero_grad()
            pv_batch        = np.zeros((batch_size, len(uq_st_pv), training_length))
            exo_batch       = np.zeros((batch_size, len(exo_train), training_length))
            batch_img       = np.zeros((batch_size, 3 * img_num, 128, 128))
            pv_Y_batch      = np.zeros((batch_size, forecast_horizon))
            gatton_pv_batch = np.zeros((batch_size, 1 , training_length * 60))

            m = 0

            for item in random_num_list :

                pv_train                 = uq_st_pv[:,item: item + training_length]
                exo_data                 = exo_train[:,item: item + training_length]
                pv_batch[m,:,:]          = pv_train
                exo_batch[m,:,:]         = exo_data
                gatton_pv_batch[m, 0, :] = gatton_pv_dict['PV'][(item) *60 :(item + training_length)*60]
                batch_img = batch_image_generation(item)

                pv_Y  = gatton_pv_dict['PV'][(item + training_length) *60 :(item + training_length + forecast_horizon)*60]
                pv_Y_batch[m, 0] = np.mean(pv_Y[-60:]) 

                m += 1

            batch_img       = numpy_to_tensor(batch_img)
            pv_batch        = numpy_to_tensor(pv_batch)   
            Y_batch         = numpy_to_tensor(pv_Y_batch)
            exo_batch       = numpy_to_tensor(exo_batch)
            gatton_pv_batch = numpy_to_tensor(gatton_pv_batch)

            pv_batch        = pv_batch.permute(0, 2, 1)
            gatton_pv_batch = gatton_pv_batch.permute(0, 2, 1)
            exo_batch       = exo_batch.permute(0, 2, 1)
            output          = model(pv_batch, gatton_pv_batch, batch_img, exo_batch)
            Y_batch         = Y_batch.squeeze(1)
            loss            = loss_fn(output , Y_batch[:,0])

            
            loss.backward()
            optimizer.step()

            running_loss_iter   += loss.item()
            running_loss_epoch  += loss.item()
            
            if k % 50 == 49: 

                logger.info("==========================") 
                logger.info(f'Epoch : [%d / %5d] Batch: [%d / %5d] Training Loss: %.3f' %(epoch + 1, num_epoch, k + 1, num_iter, running_loss_iter))  
                running_loss_iter = 0.0

            train_losslist.append(loss.item())

        logger.info("==========================") 
        logger.info(f'Epoch : [%d / %5d] Training Loss: %.3f' %(epoch + 1, num_epoch, running_loss_epoch))  

        with torch.no_grad():

            model.eval() 

            valid_length = len(valid_index_list)
            num_iter_val = int(valid_length / batch_size) -1
            valid_loss_iter  = 0.0

            for k in range(num_iter_val):
                
                random_num_list   = []
                
                for i in range(batch_size):
                    random_num_list.append(valid_index_list.pop(0))
                    
                model.train()
                optimizer.zero_grad()
                pv_batch   = np.zeros((batch_size, len(uq_st_pv), training_length))
                batch_img  = np.zeros((batch_size, 3 * img_num, 128, 128))
                pv_Y_batch = np.zeros((batch_size, forecast_horizon))
                exo_batch = np.zeros((batch_size, len(exo_train), training_length))
                gatton_pv_batch = np.zeros((batch_size, 1 , training_length * 60))

                m = 0

                for item in random_num_list :

                    pv_train = uq_st_pv[:,item: item + training_length]
                    pv_batch[m,:,:] = pv_train
                    exo_batch[m,:,:] = exo_data
                    gatton_pv_batch[m, 0, :] = gatton_pv_dict['PV'][(item) *60 :(item + training_length)*60]

                    for j in range(3, img_num):
                        x  = Image.open(fileSelectedList[(item + training_length) * 6 + j])
                        x  = transform(x)
                        batch_img[m, j * 3: j * 3 + 3, :, :] = x

                    for n in range(3):
                        x  = Image.open(fileSelectedList[(item + training_length) * 6 + j])
                        x  = transform(x)
                        y  = Image.open(fileSelectedList[(item + training_length) * 6])
                        y  = transform(y)     
                        batch_img[m, n * 3: n * 3 + 3, :, :] = x - y 


                    pv_Y  = gatton_pv_dict['PV'][(item + training_length)*60 :(item + training_length+forecast_horizon)*60]
                    pv_Y_batch[m, 0] = np.mean(pv_Y[-60:]) 

                    m += 1

                batch_img       = numpy_to_tensor(batch_img)
                pv_batch        = numpy_to_tensor(pv_batch)   
                Y_batch         = numpy_to_tensor(pv_Y_batch)
                gatton_pv_batch = numpy_to_tensor(gatton_pv_batch)
                gatton_pv_batch = gatton_pv_batch.permute(0, 2, 1)
                pv_batch        = pv_batch.permute(0, 2, 1)
                exo_batch       = numpy_to_tensor(exo_batch)
                exo_batch       = exo_batch.permute(0,2,1)
                Y_batch         = Y_batch.squeeze(1)
                output          = model(pv_batch, gatton_pv_batch, batch_img, exo_batch)

                valid_loss    = loss_fn(output , Y_batch[:,0])

                valid_loss_list.append(valid_loss.item())
                valid_loss_iter += valid_loss.item()

            logger.info("==========================") 
            logger.info(f'[%d, %5d] Validation Loss: %.3f' %(epoch, num_epoch, valid_loss_iter))  


        with open('losslist_lstm.pickle', 'wb') as handle:
            pickle.dump(train_losslist, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        with open('valid_loss_list_lstm.pickle', 'wb') as handle:
            pickle.dump(valid_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)    


    torch.save(model.state_dict(), 'BILST_en_para.pth')
    torch.save(model, 'BILST_en.pth')
    logger.info("==========================")
    logger.info("Model saved") 


logger.info("Finished") 