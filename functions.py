import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from mlxtend.data import loadlocal_mnist
import platform
import math
import struct




def encoder(input_img, convolutions, filter_size, kernel_size, dropout_size):
    #encoder
    #input 28 x 28 x 1
    conv1 = Conv2D(filter_size[0], (kernel_size[0],kernel_size[0]), activation='relu', padding='same')(input_img) 
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filter_size[1], (kernel_size[1],kernel_size[1]), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    model = MaxPooling2D(pool_size=(2, 2))(conv1)
    for i in range(2, convolutions-1, 2):
        model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) 
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[i+1], (kernel_size[i+1],kernel_size[i+1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (i == 2):
            model = MaxPooling2D(pool_size=(2, 2))(model) 
        model = Dropout(dropout_size[i+1])(model)    
    if (convolutions%2 != 0):
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)

    return model

def decoder(model, convolutions, filter_size, kernel_size, dropout_size):
    #decoder
    if (convolutions%2 != 0):
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[convolutions-2], (kernel_size[convolutions-2],kernel_size[convolutions-2]), activation='relu', padding='same')(model) 
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[convolutions-3], (kernel_size[convolutions-3],kernel_size[convolutions-3]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (convolutions==5):
            model = UpSampling2D((2,2))(model)
        model = Dropout(dropout_size[convolutions-3])(model)    

        for i in range(convolutions-4 , -1, -2):
            model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) 
            model = BatchNormalization()(model)
            model = Conv2D(filter_size[i-1], (kernel_size[i-1],kernel_size[i-1]), activation='relu', padding='same')(model)
            model = BatchNormalization()(model)
            if (i == 1 or i == 3):
                model = UpSampling2D((2,2))(model) 
            if(i != 1):
                model = Dropout(dropout_size[i-1])(model) 

        model = Conv2D(1, (kernel_size[0],kernel_size[0]), activation='sigmoid', padding='same')(model)
    else:
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model) 
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[convolutions-2], (kernel_size[convolutions-2],kernel_size[convolutions-2]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (convolutions==4):
            model = UpSampling2D((2,2))(model)
        model = Dropout(dropout_size[convolutions-1])(model)   
        for i in range(convolutions-3 , -1, -2):
            model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) 
            model = BatchNormalization()(model)
            model = Conv2D(filter_size[i-1], (kernel_size[i-1],kernel_size[i-1]), activation='relu', padding='same')(model)
            model = BatchNormalization()(model)
            if (i == 1 or i==3):
                model = UpSampling2D((2,2))(model) 
            if(i != 1):
                model = Dropout(dropout_size[i])(model)  
        
        model = Conv2D(1, (kernel_size[0],kernel_size[0]), activation='sigmoid', padding='same')(model)

    return model

def fully_connected(encode, dense_size,dropout_size):
    temp = Flatten()(encode)
    dence = Dense(dense_size, activation='relu')(temp)
    dence = BatchNormalization()(dence)
    dence = Dropout(dropout_size)(dence)
    layers = Dense(10, activation='softmax')(dence)
    return layers
