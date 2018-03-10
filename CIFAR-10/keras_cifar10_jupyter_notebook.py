# -*- coding: utf-8 -*-
"""Keras_CIFAR10_jupyter_Notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G9iw2F0Z0eBITecM9IR-704qGdrIFe7m
"""

# for google cloud gpu only
!pip install keras

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#constant
BATCH_SIZE = 128
NB_EPOCH = 25   # CPU 1, GPU 25
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

#load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

"""#### Now let's do a one-hot encoding and normalize the images:"""

# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""### Let build the network for CIFAR-10"""

# network for CIFAR-10
model = Sequential()

# Keras Con2D format
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# CONV (32 filters of size [3,3]) => RELU
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))

# max pooling with 0.25 dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# CONV (32 filters of size 3)
model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(Activation('relu'))

# max pooling with 0.25 dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
# CONV (64 filters of size 3)  
model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(Activation('relu'))

# CONV (64 filters of size 3, strides 3)
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))

# max pooling with 0.25 dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Dense layer of 512 => RELU
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

# Dense layer of NB_CLASSES=10 => softmax
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

# print model summary
model.summary()

"""### Let's train the model"""

# train it
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

"""### Evaluate it"""

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

"""### **Let's save model**"""

# save model on disk
model.save('keras_CIFAR10.h5')



"""### We need to increase training data.. 
### Data Augmentation!!
"""

# imports
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# constants
NUM_TO_AUGMENT=5

# augumenting
print("Augmenting training set images...")
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, 
                             height_shift_range=0.2, zoom_range=0.2, 
                             horizontal_flip=True,fill_mode='nearest')

datagen.fit(X_train)

"""### Let's retrain the model"""

# train
history = model.fit_generator(datagen.flow(X_train, Y_train, 
                                           batch_size=BATCH_SIZE), 
                              samples_per_epoch=X_train.shape[0], 
                              epochs=NB_EPOCH, verbose=VERBOSE)

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
