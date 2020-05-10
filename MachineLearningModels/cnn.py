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

    def __init__(self, height, width, dimension, classes, epochs=150, batch_size=50, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.Y_str2num = None

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
        data = data.reshape(len(data),self.height,self.weight,self.dimension)
        return data

    def init_model(self):
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(self.height,self.weight,self.dimension)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y.copy()

        self.X = self.datareshape(self.X)
        self.Y = self.map_str_to_number(self.Y)

        print('Convolutional Neural Network Train started............')
        self.model.fit(self.X.values, self.Y.values, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_split=0.2)
        print('Convolutional Neural Network Train completed..........')

        return self.model

    def save(self, filename='cnn_model.h5'):
        if not os.path.exists(filename):
            os.mknod(filename)
        self.model.save(filename)

    def predict(self, test_X):
        test_X = self.datareshape(test_X)
        predictions = self.model.predict(test_X.values)
        predictions = predictions.round()
        return predictions[:,0]

    def score(self, test_X, test_Y):
        y_pred = self.predict(test_X)
        r2s = r2_score(test_Y, y_pred, multioutput='variance_weighted')
        return r2s

    def map_str_to_number(self, Y):
        if self.mapping_dict is not None:
            for label_header in self.label_headers:
                Y[label_header] = Y[label_header].map(self.mapping_dict)
            return Y

        mapping_dict = None
        for label_header in self.label_headers:
            check_list = pd.Series(Y[label_header])
            for item in check_list:
                if type(item) == str:
                    mapping_flag = True
                    break
            if mapping_flag:
                classes = Y[label_header].unique()
                mapping_dict = {}
                index = 0
                for c in classes:
                    mapping_dict[c] = index
                    index += 1

                Y[label_header] = Y[label_header].map(mapping_dict)
                mapping_flag = False

        self.mapping_dict = mapping_dict
        return Y

    def map_number_to_str(self, Y, classes):
        if self.mapping_dict is not None:
            mapping_dict = self.mapping_dict
        else:
            Y = Y.round()
            mapping_dict = {}
            index = 0
            for c in classes:
                mapping_dict[index] = c
                index += 1
        return Y.map(mapping_dict)

    def getAccuracy(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = self.map_str_to_number(test_labels.copy())

        correct = 0
        for i in range(len(df)):
            if (df.values[i] == test_labels.values[i]):
                correct = correct + 1
        return correct/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'classifier':
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                df_tmp = self.map_number_to_str(df.ix[:,index], classes)
                title = 'Normalized confusion matrix for ConvolutionalNeuralNetwork (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df_tmp, classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
            return 'No Confusion Matrix for Regression'

    def featureImportance(self):
        return 'No feature importance for CNN'

    def getRSquare(self, test_labels, predictions, mode='single'):
        return 'No RSquare for Classification'

    def getMSE(self, test_labels, predictions):
        return 'No MSE for Classification'

    def getMAPE(self, test_labels, predictions):
        return 'No MAPE for Classification'

    def getRMSE(self, test_labels, predictions):
        return 'No RMSE for Classification'

    def load(self, path, type):
        self.model = load_model(path)
        self.type = type
        return self.model
