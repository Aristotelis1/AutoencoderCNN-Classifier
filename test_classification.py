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
from sklearn.metrics import classification_report

import argparse
from mlxtend.data import loadlocal_mnist
import platform

from keras.models import load_model
import math

import random

from functions import encoder, fully_connected


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
    part = input("Type 'part' to use a part of your dataset: ")
    if(part == "part"):
        part = input("Type the number of images you want to train: ")
        part = int(part)
        if(part>number_of_images_train):
            print('You typed more images than expected. We will work with whole dataset.')
        else:
            number_of_images_train = part
    dimensions = int(X.shape[1])

    X1,Y1 = loadlocal_mnist(
        images_path=args.test_set,
        labels_path=args.test_labels)
        
    number_of_images_test = int(X1.shape[0])
    part = input("Type 'part' to use a part of your testset: ")
    if(part == "part"):
        part = input("Type the number of images you want to test: ")
        part = int(part)
        if(part>number_of_images_test):
            print('You typed more images than expected. We will work with whole testset.')
        else:
            number_of_images_test = part

    print('%s images with dimension: %s' % (X.shape[0],X.shape[1]))
    print('labels: %s' % np.unique(Y))



    model=load_model(args.model)       
    weights=model.get_weights()

    history_list = []
    prediction_list = []
    while(1):
        x, y = int(math.sqrt(dimensions)), int(math.sqrt(dimensions))
        CNN_convs = 8
        filters_size_list = [32,32,64,64,128,128,256,256]
        kernel_size_list = [3,3,3,3,3,3,3,3]
        dropout_list = [0,0.70,0,0.70,0,0.70,0,0.70]
        fc_dense_size = 128
        fc_dropout = 0.7
        create = input("Type 'create' if you want to create your own model: ")
        if(create == 'create'):
            filters_size_list.clear()
            kernel_size_list.clear()
            dropout_list.clear()
            print('###################################################################################################################')
            print("The CNN model has blocks of: 2 convolutions, each followed by 1 BatchNormalization and after that, \n1 Dropout layer, except the first block which doesnt have Dropout after it.")
            print("The first 2 blocks are also followed by 2 downsampling layers, so that the final image-layer has shape %s x %s" %(x/4 , y/4))
            print('###################################################################################################################')
            CNN_convs = input ("Type the number of convolutions you want: ")
            CNN_convs = int(CNN_convs)
            while(CNN_convs < 3):
                CNN_convs = input ("Type the number of convolutions you want (at least 4 needed): ")
                CNN_convs = int(CNN_convs)

            for i in range(0, CNN_convs):
                filter_size = input ("Type the number of filter in Convolution(%d): " %(i+1))
                filters_size_list.append( int(filter_size) )
                kernel_size = input ("Type the kernel size of Convolution(%d): " %(i+1))
                kernel_size_list.append( int(kernel_size) )
                if (i%2 == 1 and i!=1):
                    dropout_size = input ("Type the dropout size (0.xx) after Convolution(%d): " %(i+1))
                    dropout_list.append( float(dropout_size))
                else:
                    dropout_list.append(0)
            fc_dense_size = input("Type the number of neurons in fully connected part: ")
            fc_dropout = input("Type the dropout size (0.xx) in fully connected part: ")
            fc_dense_size = int(fc_dense_size)
            fc_dropout = float(fc_dropout)
        inChannel =  input("inChannel: ")
        batch_size = input("Batch Size: ")
        epochsenc = input("Epochs for fully connected part: ")
        epochs = input("Epochs for whole model: ")
        inChannel = int(inChannel)
        batch_size = int(batch_size)
        epochs = int(epochs)
        epochsenc = int(epochsenc)
        input_img =  Input(shape=(x, y, inChannel), name='input')

        train_X = np.reshape(X, (len(X), 28, 28, 1))
        train_X = train_X.astype('float32')
        train_X = train_X/255.0

        train_X = train_X[:number_of_images_train]
        Y = Y[:number_of_images_train]

        test_X = np.reshape(X1, (len(X1), 28, 28, 1))
        test_X = test_X.astype('float32')
        test_X = test_X/255.0

        test_X = test_X[:number_of_images_test]
        Y1 = Y1[:number_of_images_test]

        train_Y_one_hot = to_categorical(Y)
        test_Y_one_hot = to_categorical(Y1)

        encode = encoder(input_img,CNN_convs, filters_size_list, kernel_size_list, dropout_list)
        fc_model = Model(input_img, fully_connected(encode, fc_dense_size,fc_dropout))
        enco_layers = len(fc_model.layers)-5    #get number of layers on encoder
        trained = input("Type 'pre' to use pretrained model: ")
        if(trained == 'pre'):       #option to use pretrained whole model
            model_path = input("Give me the path to pretrained model: ")
            model = load_model(model_path)
            weights = model.get_weights()
            for m1, m2 in zip(fc_model.layers[:], model.layers[:]):
                m1.set_weights(m2.get_weights())
        else:   #default option to use pretrained autoencoder model
            for m1, m2 in zip(fc_model.layers[:enco_layers], model.layers[0:enco_layers]):
                m1.set_weights(m2.get_weights())

        #train only fully connected layer
        for x in fc_model.layers[0:enco_layers]:
            x.trainable=False
        fc_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
        #fc_model.summary() #uncomment to see the summary of the model
        train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size=0.20,random_state=13)   #splitting persentage can change
        fc_train = fc_model.fit(train_X, train_label, batch_size=batch_size ,epochs=epochsenc,verbose=1,validation_data=(valid_X, valid_label))

        #train whole model (all layers including already trained)
        for x in fc_model.layers[:enco_layers]:
            x.trainable=True
        fc_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])
        fc_train = fc_model.fit(train_X, train_label, batch_size=batch_size ,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
        filters = (','.join(str(x) for x in filters_size_list))
        kernel = (','.join(str(x) for x in kernel_size_list))
        drops = (','.join(str(x) for x in dropout_list))

        history_list.append((fc_train,batch_size,inChannel,epochsenc, epochs, CNN_convs, filters, kernel, drops, fc_dense_size, fc_dropout))
        predicted_classes = fc_model.predict(test_X)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

        test_eval = fc_model.evaluate(test_X, test_Y_one_hot, verbose=0)


        prediction_list.append((predicted_classes,test_eval))
        #show plot datagrams
        pl = input("Type 'yes' to plot: ")
        if(pl == 'yes'):
            for history,predict in zip(history_list,prediction_list):
                print('Batches: %d\nEpochs (fully connected): %d\nEpochs (whole model): %d\nConvolutions: %d\nFully connected neurons: %d' %(history[1], history[3], history[4], history[5], history[9]))
                target_names = ["Number {}".format(i) for i in range(10)]
                print('Test loss:', predict[1][0])
                print('Test accuracy:', predict[1][1])
                print(classification_report(Y1, predict[0], target_names=target_names))
                accuracy = history[0].history['accuracy']
                val_accuracy = history[0].history['val_accuracy']
                loss = history[0].history['loss']
                val_loss = history[0].history['val_loss']
                epochs = range(len(accuracy))
                plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
                plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
                plt.title('Filters: %s\nKernel size: %s\nDropout(encoder): %s\nNeurons FC: %d\nDropout(fc): %s' %(history[6], history[7], history[8], history[9], history[10]), loc='left')
                plt.title('Convolutions: %d\nBatches: %d\ninChannell: %d\nEpochs for fc: %d\nEpochs for model: %d' %(history[5], history[1], history[2], history[3], history[4]), loc='right')
                plt.xlabel('Epochs (whole model)')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.figure()
                plt.plot(epochs, loss, 'bo', label='Training loss')
                plt.plot(epochs, val_loss, 'b', label='Validation loss')
                plt.title('Filters: %s\nKernel size: %s\nDropout(encoder): %s\nNeurons FC: %d\nDropout(fc): %s' %(history[6], history[7], history[8], history[9], history[10]), loc='left')
                plt.title('Convolutions: %d\nBatches: %d\ninChannell: %d\nEpochs for fc: %d\nEpochs for model: %d' %(history[5], history[1], history[2], history[3], history[4]), loc='right')
                plt.xlabel('Epochs (whole model)')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

        next_move = input("Type 'print' to print: ")
        #show some images with their prediction
        if(next_move == 'print'):
            flag = False
            while(flag == False):
                ubatch_size = input("Batch Size: ") #128 stis diafaneies
                uepochsenc = input("Epochs for fully connected part: ")
                uepochs = input("Epochs for whole model: ")
                pre = input("Type 'yes' if you didn't create model: ")
                if (pre == "yes"):
                    uconvs=8
                    ufc_neurons=128
                else:
                    uconvs = input("Convolutions on encoder part: ")
                    ufc_neurons = input("Neurons on fully connected part: ")
                    ufc_neurons = int(ufc_neurons)
                    uconvs = int(uconvs)    
                uepochsenc = int(uepochsenc)
                ubatch_size = int(ubatch_size)
                uepochs = int(uepochs)
                for history,predict in zip(history_list,prediction_list):
                    if(history[1] == ubatch_size and history[3] == uepochsenc and history[4] == uepochs and history[5] == uconvs and history[9] == ufc_neurons):
                        correct = np.where(predict[0]==Y1)[0]
                        print ("Found %d correct labels" % len(correct))
                        incorrect = np.where(predict[0] !=Y1)[0]
                        print ("Found %d incorrect labels" % len(incorrect))

                        for i, num in enumerate(test_X[:16]):
                            r = random.randint(0,number_of_images_test)
                            plt.subplot(4,4,i+1)
                            plt.imshow(test_X[r].reshape(28,28), cmap='gray', interpolation='none')
                            plt.title("Predicted {}, Label {}".format(predict[0][r], Y1[r]))
                            plt.tight_layout()
                        plt.show()
                        flag = True
                if(flag == False):
                    print("Not found")
                    answer = input("You want to try again? Type 'yes': ")
                    if(answer != 'yes'):
                        break
        #save whole model
        next_move = input("Type 'save' to save: ")
        if(next_move == 'save'):
            path = input("Give me the path to save the previous fully connected model: ")
            fc_model.save(path)

        next_move = input("Type '0' to stop: ")
        if(next_move == '0'):
            exit()
