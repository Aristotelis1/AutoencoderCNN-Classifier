import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input, BatchNormalization
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import argparse
from mlxtend.data import loadlocal_mnist
import platform

from keras.models import load_model
import math

def encoder(input_img, filters):
    #encoder
    #input 28 x 28 x 1

    conv1 = Conv2D(filters, (3,3), activation='relu', padding='same')(input_img) # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 14 x 14 x 32
    filters=filters*2
    conv2 = Conv2D(filters, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    filters=filters*2
    conv3 = Conv2D(filters, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small & thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    filters=filters*2
    conv4 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small & thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fully_connected(encode, filters):
    temp = Flatten()(encode)
    dence = Dense(128, activation='relu')(temp)
    layers = Dense(10, activation='softmax')(dence)
    return layers

if __name__ == "__main__":

    # reading the mnist files
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_set", "-d", type=str, required=True)
    parser.add_argument("--train_labels", "-dl", type=str, required=True)
    parser.add_argument("--test_set", "-t", type=str, required=True)
    parser.add_argument("--test_labels", "-tl", type=str, required=True)
    parser.add_argument("--model", "-model", type=str, required=True)
    args = parser.parse_args()
    print(args.train_set)
    print(args.train_labels)


    X,Y = loadlocal_mnist(
        images_path=args.train_set,
        labels_path=args.train_labels)
        
    number_of_images_train = int(X.shape[0])
    dimensions = int(X.shape[1])

    X1,Y1 = loadlocal_mnist(
        images_path=args.test_set,
        labels_path=args.test_labels)
        
    number_of_images_test = int(X1.shape[0])


    print('Dimensions: %s x %s' % (X.shape[0],X.shape[1]))
    print('Digits:  0 1 2 3 4 5 6 7 8 9')
    print('labels: %s' % np.unique(Y))


    train_X = np.reshape(X, (len(X), 28, 28, 1))
    train_X = train_X.astype('float32')
    train_X = train_X/255.0

    test_X = np.reshape(X1, (len(X1), 28, 28, 1))
    test_X = test_X.astype('float32')
    test_X = test_X/255.0

    # model=Model.create_model()
    # model.evaluate(test_X, y1)
    # model.load_weights(args.model)

    model=load_model(args.model)
    weights=model.get_weights()
#    model.summary()
    print(len(weights))

    x, y = int(math.sqrt(dimensions)), int(math.sqrt(dimensions))
    inChannel =  input("inChannel: ")
    batch_size = input("Batch Size: ") #128 stis diafaneies
    epochs = input("Epochs: ")
    inChannel = int(inChannel)
    batch_size = int(batch_size)
    epochs = int(epochs)
    filters = input("Give me the number of filters: ")
    filters = int(filters)
    input_img =  Input(shape=(x, y, inChannel), name='input')

    train_Y_one_hot = to_categorical(Y)
    test_Y_one_hot = to_categorical(Y1)

    encode = encoder(input_img, filters)
    fc_model = Model(input_img, fully_connected(encode, filters))
    for m1, m2 in zip(fc_model.layers[:19], model.layers[0:19]):
        m1.set_weights(m2.get_weights())
    for x in fc_model.layers[0:19]:
        x.trainable=False
    fc_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size=0.2,random_state=13)

    fc_train = fc_model.fit(train_X, train_label, batch_size=batch_size ,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


