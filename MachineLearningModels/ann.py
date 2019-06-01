import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from MachineLearningModels.model import Model
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
import os


class NeuralNetwork(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    path = 'models/nn.h5'


    def __init__(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = Sequential()

    def add(self, layer, input_dim=None, kernel_initializer=None, activation='linear'):
        if (kernel_initializer == None and input_dim == None):
            self.model.add(Dense(layer, activation=activation))
        else:
            self.model.add(Dense(layer, input_dim=input_dim, kernel_initializer=kernel_initializer, activation=activation))

    def summary(self):
        self.model.summary()

    def compile(self, loss='mse', optimizer='adam', metrics=['mse','mae']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X=None, Y=None, epochs=150, batch_size=50,  verbose=1, validation_split=0.2):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Neural Network Train started............')
        self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size,  verbose=verbose, validation_split=validation_split)
        print('Neural Network Train completed..........')

        return self.model

    def save(self, path=None):
        if path is not None:
            self.path = path

        if not os.path.exists('models'):
            os.mkdir('models')

        if not os.path.exists(self.path):
            os.mknod(self.path)

        self.model.save(self.path)

    def predict(self, test_X):
        return self.model.predict(test_X)

    def score(self, test_X, test_Y):
        y_pred = self.predict(test_X)
        r2s = r2_score(test_Y, y_pred, multioutput='variance_weighted')
        return r2s

    def load(self, path):
         self.model = load_model(path)
         return self.model
