import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
import matplotlib.pyplot as plt

import argparse
from mlxtend.data import loadlocal_mnist
import platform



if __name__ == "__main__":
    #test_files
    test_X, test_y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')
    
    test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))

    reconstructed_model = keras.models.load_model('autoencoder')

    print(test_X[0].shape)
    print(test_X[0])
    test_X = test_X.astype('float32')
    test_X = test_X/255.0
    print(test_X[0])
    test_X = test_X[:10] #uncomment this for less test images
    out_images = reconstructed_model.predict(test_X)

    #Plot the generated data
    n=10 #how many digits we will display
    plt.figure(figsize=(15, 4))
    for i in range(n):
        #display original
        ax = plt.subplot(2, n , i+1)
        plt.imshow(test_X[i].reshape(28, 28))

        #display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(out_images[i].reshape(28, 28))
    
    plt.show()