import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from MachineLearningModels.model import Model
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import load_model
import os
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
from keras.utils import to_categorical



class ConvolutionalNeuralNetwork(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    path = 'models/cnn.h5'
    regressor = None
    classifier = None

    def __init__(self):
        Model.__init__(self)

    def __init__(self, height, width, dimension, classes, label_headers, epochs=150, batch_size=50, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.Y_str2num = None
        self.label_headers = label_headers

        self.mapping_dict = None

        self.type = 'classifier'

        self.height = height
        self.width = width
        self.dimension = dimension
        self.classes = classes

        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential()

        self.init_model()

    def datareshape(self, data):
        data = data.reshape(len(data),self.height,self.width,self.dimension)
        return data

    def init_model(self):
        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(self.height,self.width,self.dimension)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='linear'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y.copy()

        self.X = self.datareshape(self.X)
        self.Y = to_categorical(self.Y, num_classes=self.classes)

        print('Convolutional Neural Network Train started............')
        self.model.fit(self.X, self.Y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_split=0.2)
        print('Convolutional Neural Network Train completed..........')

        return self.model

    def save(self, filename='cnn_model.h5'):
        self.model.save(filename)

    def predict(self, test_X):
        test_X = self.datareshape(test_X)
        predictions = self.model.predict(test_X)
        predictions = predictions.round()
        return predictions[:,0]

    def score(self, test_X, test_Y):
        test_Y = to_categorical(test_Y, num_classes=self.classes)
        score = self.model.evaluate(test_X, test_Y, batch_size=32)
        return score

    def load(self, path, type):
        self.model = load_model(path)
        self.type = type
        return self.model
