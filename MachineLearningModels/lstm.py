import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM as lstm_model
from MachineLearningModels.model import Model
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class LSTMModel(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    path = 'models/lstm.h5'
    regressor = None
    classifier = None

    def __init__(self):
        Model.__init__(self)

    def __init__(self, feature_headers, label_headers, type='regressor', epochs=150, batch_size=50, lookback=1, num_of_cells=4, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.feature_headers = feature_headers
        self.label_headers = label_headers

        self.no_inputs = len(feature_headers)
        self.no_outputs = len(label_headers)

        self.epochs = epochs
        self.batch_size = batch_size

        self.lookback = lookback
        self.num_of_cells = num_of_cells

        self.type = type
        self.model = Sequential()
        self.init_model()

    def init_model(self):
        self.model.add(lstm_model(self.num_of_cells, input_shape=(self.lookback,self.no_inputs)))
        if self.type == 'regressor':
            output_activation = 'linear'
        else:
            output_activation = 'tanh'
        self.model.add(Dense(self.no_outputs, activation=output_activation))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

    def summary(self):
        self.model.summary()

    def compile(self, loss='mae', optimizer='adam', metrics=['mse','mae']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def create_dataset(self, X, Y=None):
        dataX = []
        dataY = []
        for i in range(len(X) - self.lookback - 1):
            a = X.values[i:(i+self.lookback), :]
            dataX.append(a)
            if Y is not None:
                dataY.append(Y.values[i+self.lookback, :])
        if Y is not None:
            return dataX, dataY
        else:
            return dataX

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y.copy()

        dataX, dataY = self.create_dataset(self.X, self.Y)
        dataX, trainY= np.array(dataX), np.array(dataY)
        trainX = np.reshape(dataX, (dataX.shape[0],self.lookback, self.no_inputs))

        mapping_flag = False
        if self.type == 'classifier':
            self.Y = self.map_str_to_number(self.Y)
        print('LSTM Train started............')
        self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_split=0.2)
        print('LSTM Train completed..........')

        return self.model

    def save(self, filename='lstm_model.h5'):
        self.model.save(filename)

    def predict(self, test_X):
        dataX = self.create_dataset(test_X)
        dataX = np.array(dataX)
        testX = np.reshape(dataX, (dataX.shape[0], self.lookback, self.no_inputs))

        predictions = self.model.predict(testX)
        if self.type == 'classifier':
            predictions = predictions.round()
        return predictions[:,0]

    def score(self, test_X, test_Y):
        if test_Y is None:
            test_Y = self.create_dataset(test_X)
        y_pred = self.predict(test_X)
        r2s = r2_score(test_Y, y_pred, multioutput='variance_weighted')
        return r2s

    def map_str_to_number(self, Y):
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
        return Y

    def map_number_to_str(self, Y, classes):
        Y = Y.round()
        mapping_dict = {}
        index = 0
        for c in classes:
            mapping_dict[index] = c
            index += 1
        return Y.map(mapping_dict)

    def getAccuracy(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]

        if self.type == 'classifier':
            test_labels = self.map_str_to_number(test_labels.copy())

        correct = 0
        for i in range(len(df)):
            if (df.values[i] == test_labels.values[i]):
                correct = correct + 1
        return correct/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]
        if self.type == 'classifier':
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                df_tmp = self.map_number_to_str(df.ix[:,index], classes)
                title = 'Normalized confusion matrix for LSTM (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df_tmp, classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
            return 'No Confusion Matrix for Regression'

    def featureImportance(self):
        return 'No feature importance for LSTM'

    def getRSquare(self, test_labels, predictions, mode='single'):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]
        if self.type == 'regressor':
            if mode == 'multiple':
                errors = r2_score(test_labels, df, multioutput='variance_weighted')
            else:
                errors = r2_score(test_labels, df)
            return errors
        else:
            return 'No RSquare for Classification'

    def getMSE(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]
        if self.type == 'regressor':
            errors = mean_squared_error(test_labels, df)
            return errors
        else:
            return 'No MSE for Classification'

    def getMAPE(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]
        if self.type == 'regressor':
            errors = np.mean(np.abs((test_labels - df.values) / test_labels)) * 100
            return errors.values[0]
        else:
            return 'No MAPE for Classification'

    def getRMSE(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        test_labels = test_labels[(self.lookback+1):]
        if self.type == 'regressor':
            errors = sqrt(mean_squared_error(test_labels, df))
            return errors
        else:
            return 'No RMSE for Classification'

    def load(self, path, type):
        self.model = load_model(path)
        self.type = type
        return self.model
