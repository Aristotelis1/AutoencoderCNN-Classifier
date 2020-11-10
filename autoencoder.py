import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import argparse
from mlxtend.data import loadlocal_mnist
import platform



def encoder(input_img):
    #encoder
    #input 28 x 28 x 1

    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_img) # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small & thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small & thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


def preprocess(x):
    x = x.astype('float32') / 255.
    return x.reshape(-1, np.prod(x.shape[1:])) #flatten


if __name__ == "__main__":

    # reading the mnist files
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-d", type=str, required=True)
    args = parser.parse_args() 
    # Dataset is save in args.file

    #train_files
    X, y = loadlocal_mnist(
        images_path=args.file,
        labels_path='train-labels-idx1-ubyte')
    
    print('Dimensions: %s x %s' % (X.shape[0],X.shape[1]))

    #np.savetxt(fname='images.csv',X=X, delimiter=',',fmt="%d") #uncomment to save it in csv file

    #test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))
    #print(train_X)

    # Define the convolutional Autoencoder Model
    x, y = 28, 28
    inChannel = 1
    batch_size = 128 #128 stis diafaneies
    epochs = 50
    input_img =  Input(shape=(x, y, inChannel), name='input')


    train_X = np.reshape(X, (len(X), 28, 28, 1))
    print(train_X[0].shape)
    print(train_X[0])
    train_X = train_X.astype('float32')
    train_X = train_X/255.0
    print(train_X[0])
    
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer = "adam") #RMSprop() stis diafaneies
    #autoencoder.summary() #uncomment to see the summary of the AE

    #train_X = preprocess(train_X)
    #train_X, valid_X = train_test_split(X,test_size=0.2, random_state=13)
    #print('Dimensions: %s x %s' % (train_X.shape[0],train_X.shape[1]))

    train_X = train_X[:2000] #uncomment this in order to train with less images
    autoencoder_train = autoencoder.fit(train_X, train_X, batch_size = batch_size,epochs = epochs,verbose=1)

    autoencoder.save('autoencoder')








