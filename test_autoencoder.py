import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels


import argparse
from mlxtend.data import loadlocal_mnist
import platform
import math
import struct



def encoder(input_img, convolutions, filter_size, kernel_size, dropout_size):
    #encoder
    #input 28 x 28 x 1
    conv1 = Conv2D(filter_size[0], (kernel_size[0],kernel_size[0]), activation='relu', padding='same')(input_img) # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filter_size[1], (kernel_size[1],kernel_size[1]), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 14 x 14 x 32
    model = Dropout(dropout_size[1])(pool1)    #drop1

    for i in range(2, convolutions, 2):
        model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) # 28 x 28 x 32
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[i+1], (kernel_size[i+1],kernel_size[i+1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (i == 2):
            model = MaxPooling2D(pool_size=(2, 2))(model) # 14 x 14 x 32
        if (i!=convolutions-1 and i!=convolutions-2):
            model = Dropout(dropout_size[i+1])(model)    #drop1

    if (convolutions%2 != 0):
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)

    return model

def decoder(model, convolutions, filter_size, kernel_size, dropout_size):
    #decoder
    if (convolutions%2 != 0):
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (convolutions==5):
            model = UpSampling2D((2,2))(model)
        model = Conv2D(filter_size[convolutions-2], (kernel_size[convolutions-2],kernel_size[convolutions-2]), activation='relu', padding='same')(model) # 28 x 28 x 32
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[convolutions-3], (kernel_size[convolutions-3],kernel_size[convolutions-3]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (convolutions==5):
            model = UpSampling2D((2,2))(model)
        model = Dropout(dropout_size[convolutions-3])(model)    #drop1

        for i in range(convolutions-4 , -1, -2):
            model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) # 28 x 28 x 32
            model = BatchNormalization()(model)
            model = Conv2D(filter_size[i-1], (kernel_size[i-1],kernel_size[i-1]), activation='relu', padding='same')(model)
            model = BatchNormalization()(model)
            if (i == 1 or i==3):
                model = UpSampling2D((2,2))(model) # 14 x 14 x 32
            model = Dropout(dropout_size[i-1])(model)    #drop1
    else:
        model = Conv2D(filter_size[convolutions-1], (kernel_size[convolutions-1],kernel_size[convolutions-1]), activation='relu', padding='same')(model) # 28 x 28 x 32
        model = BatchNormalization()(model)
        model = Conv2D(filter_size[convolutions-2], (kernel_size[convolutions-2],kernel_size[convolutions-2]), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        if (convolutions==4):
            model = UpSampling2D((2,2))(model)

        for i in range(convolutions-3 , -1, -2):
            model = Conv2D(filter_size[i], (kernel_size[i],kernel_size[i]), activation='relu', padding='same')(model) # 28 x 28 x 32
            model = BatchNormalization()(model)
            model = Conv2D(filter_size[i-1], (kernel_size[i-1],kernel_size[i-1]), activation='relu', padding='same')(model)
            model = BatchNormalization()(model)
            if (i == 0 or i==2):
                model = UpSampling2D((2,2))(model) # 14 x 14 x 32
            model = Dropout(dropout_size[i-1])(model)    #drop1

    return model


if __name__ == "__main__":

    # reading the mnist files
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-d", type=str, required=True)
    args = parser.parse_args() 
    # Dataset is save in args.file

    #train_files
    # X, y = loadlocal_mnist(
    #     images_path=args.file,
    #     labels_path='train-labels-idx1-ubyte')
    
    with open(args.file, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        X = np.fromfile(imgpath,dtype=np.uint8).reshape(num, 784)
    
    print('Dimensions: %s x %s' % (X.shape[0],X.shape[1]))
    number_of_images = int(X.shape[0])
    dimensions = int(X.shape[1])
    part = input("Type 'part' to use a part of your dataset: ")
    if(part == "part"):
        part = input("Type the number of images you want to train: ")
        part = int(part)
        if(part>number_of_images):
            print('You typed more images than expected. We will work with whole dataset.')
        else:
            number_of_images = part
    #np.savetxt(fname='images.csv',X=X, delimiter=',',fmt="%d") #uncomment to save it in csv file

    #test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))
    #print(train_X)

    history_list = []

    while(1):

        # Define the convolutional Autoencoder Model
        x, y = int(math.sqrt(dimensions)), int(math.sqrt(dimensions))
        print("The CNN model has blocks of: 2 convolutions, each followed by 1 BatchNormalization and after that, 1 Dropout layer, except the last block which doesnt have Dropout after it.")
        print("The first 2 blocks are also followed by 2 downsampling layers, so that the final image-layer has shape %s x %s" %(x/4 , y/4))
        CNN_convs = input ("Type the number of convolutions you want: ")
        CNN_convs = int(CNN_convs)
        while(CNN_convs < 3):
            CNN_convs = input ("Type the number of convolutions you want: ")
            CNN_convs = int(CNN_convs)
        filters_size_list = []
        kernel_size_list = []
        dropout_list = []
        for i in range(0, CNN_convs):
            filter_size = input ("Type the number of filter in Convolution(%d): " %(i+1))
            filters_size_list.append( int(filter_size) )
            kernel_size = input ("Type the kernel size of Convolution(%d): " %(i+1))
            kernel_size_list.append( int(kernel_size) )
            if (i%2 == 1 and i!=CNN_convs-1):
                dropout_size = input ("Type the dropout size (0.xx) after Convolution(%d): " %(i+1))
                dropout_list.append( float(dropout_size))
            else:
                dropout_list.append(0)


        inChannel =  input("inChannel: ")
        batch_size = input("Batch Size: ") #128 stis diafaneies
        epochs = input("Epochs: ")
        inChannel = int(inChannel)
        batch_size = int(batch_size)
        epochs = int(epochs)
        input_img =  Input(shape=(x, y, inChannel), name='input')

        train_X = np.reshape(X, (len(X), 28, 28, 1))
        train_X = train_X.astype('float32')
        train_X = train_X/255.0
        autoencoder = Model(input_img, decoder(encoder(input_img,CNN_convs, filters_size_list, kernel_size_list, dropout_list), CNN_convs, filters_size_list, kernel_size_list, dropout_list))
        trained = input("Type 'pre' to use pretrained model: ")
        if(trained == 'pre'):       #option to use pretrained model
            model_path = input("Give me the path to pretrained model: ")
            model = load_model(model_path)
            weights = model.get_weights()
            for m1, m2 in zip(autoencoder.layers[:], model.layers[:]):
                m1.set_weights(m2.get_weights())
        autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) #RMSprop() stis diafaneies
        #autoencoder.summary() #uncomment to see the summary of the AE

        #train_X = preprocess(train_X)
        train_X = train_X[:number_of_images] # uncomment if you want less images
        train_X, valid_X, train_ground, valid_ground = train_test_split(train_X,train_X,test_size=0.2, random_state=13)
        #print('Dimensions: %s x %s' % (train_X.shape[0],train_X.shape[1]))

        history = autoencoder.fit(train_X, train_ground, batch_size = batch_size,epochs = epochs,verbose=1, validation_data=(valid_X,valid_ground))
        history_list.append((history,batch_size,inChannel,epochs,32))
        pl = input("Type 'yes' to plot: ")
        if(pl == 'yes'):

            for history in history_list:
                #summarize history for loss
                plt.plot(history[0].history['loss'], label= 'train loss')
                plt.plot(history[0].history['val_loss'], label= 'val loss')
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epochs')
    #            plt.legend(['train_loss', 'val_loss'], loc='upper left')
                plt.title('Batches: %d\ninChannell: %d\nEpochs: %d\nFilters: %d' %(history[1], history[2], history[3], history[4]), loc='left')

                plt.legend()
                plt.show()
        
        next_move = input("Type 'save' to save: ")
        if(next_move == 'save'):
            path = input("Give me the path to save the previous autoencoder model: ")
            autoencoder.save(path)

        next_move = input("Type '0' to stop: ")
        if(next_move == '0'):
            quit()








