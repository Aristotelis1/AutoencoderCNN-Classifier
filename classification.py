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


if __name__ == "__main__":

    # reading the mnist files
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_set", "-d", type=str, required=True)
    parser.add_argument("--train_labels", "-dl", type=str, required=True)
    parser.add_argument("--test_labels", "-t", type=str, required=True)
    parser.add_argument("--model", "-model", type=str, required=True)
    args = parser.parse_args()

    print(args.train_set)
    print(args.train_labels)