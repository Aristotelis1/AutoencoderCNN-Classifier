# AutoencoderNN-Classifier

## Train your autoencoder:
$python test_autoencoder.py –d <#dataset>

### Default Autoencoder layer-block:

-----functions.py--------

Encoder:
Convolutional layer followed by Batch Normalizaiton. 
After the second normalization, there is a downsampling (Maxpooling).
Dropout layers have been added after the second layer-block.
Decoder:
Decoder part is a mirror of the encoder with a sigmoid as an activation function for the last convolutional layer.

## Train your classification model:
$python test_classification.py –d <#training set> –dl <#training labels> -t <#testset> -tl <#test labels> -model <#autoencoder model>

Taking the encoder part of the autoencoder and adds the following layers:
->Flatten layer
->Dense layer
->Batch Normalization
->Dropout layer
->Dense(units = 10)

Accuracy: 0,9952
