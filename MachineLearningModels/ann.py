import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from MachineLearningModels.model import Model
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    path = 'models/nn.h5'
    regressor = None
    classifier = None

    def __init__(self):
        Model.__init__(self)

    def __init__(self, feature_headers, label_headers, type='regressor', epochs=150, batch_size=50, X=None, Y=None):
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

        self.mapping_dict = None

        self.type = type
        self.model = Sequential()
        self.init_model()

    def init_model(self):
        self.model.add(Dense(self.no_inputs*2+1, input_dim=self.no_inputs, kernel_initializer='normal', activation='sigmoid'))
        self.model.add(Dense(self.no_outputs, activation='sigmoid'))
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

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y.copy()

        if self.type == 'classifier':
            self.Y = self.map_str_to_number(self.Y)

        print('Neural Network Train started............')
        self.model.fit(self.X.values, self.Y.values, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_split=0.2)
        print('Neural Network Train completed..........')

        return self.model

    def save(self, filename='nn_model_test.h5'):
        self.model.save(filename)

    def predict(self, test_X):
        predictions = self.model.predict(test_X.values)
        if self.type == 'classifier':
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
        Y = Y.round()
        Y = Y.astype(int)
        if self.mapping_dict is not None:
            mapping_dict = self.mapping_dict
        else:
            mapping_dict = {}
            index = 0
            for c in classes:
                mapping_dict[index] = c
                index += 1

        inv_map = {v: k for k, v in mapping_dict.items()}
        return Y.map(inv_map)

    def getAccuracy(self, test_labels, predictions, origin=0, hitmissr=0.8):
        if self.type == 'classifier':
            correct = 0
            df = pd.DataFrame(data=predictions.flatten())
            test_labels = self.map_str_to_number(test_labels.copy())
            for i in range(len(df)):
                if (df.values[i] == test_labels.values[i]):
                    correct = correct + 1
        else:
            correct = 0
            df = pd.DataFrame(data=predictions.flatten())
            for i in range(len(df)):
                if 1 - abs(df.values[i] - test_labels.values[i])/abs(df.values[i]) >= hitmissr:
                    correct = correct + 1
        return float(correct)/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'classifier':
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                df_tmp = self.map_number_to_str(df.ix[:,index], classes)
                title = 'Normalized confusion matrix for NeuralNetwork (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df_tmp, classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
            return 'No Confusion Matrix for Regression'

    def getROC(self, test_labels, predictions, label_headers):
        predictions=pd.DataFrame(data=predictions.flatten())
        predictions.columns=test_labels.columns.values
        if self.type == 'classifier':
            test_labels = self.map_str_to_number(test_labels)
            fpr, tpr, _ = roc_curve(test_labels, predictions)
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.show()
        else:
            return 'No Confusion Matrix for Regression'

    def featureImportance(self):
        return ''

    def getRSquare(self, test_labels, predictions, mode='single'):
        df = pd.DataFrame(data=predictions.flatten())
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
        if self.type == 'regressor':
            errors = mean_squared_error(test_labels, df)
            return errors
        else:
            return 'No MSE for Classification'

    def getMAPE(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'regressor':
            errors = np.mean(np.abs((test_labels - df.values) / test_labels)) * 100
            return errors.values[0]
        else:
            return 'No MAPE for Classification'

    def getRMSE(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'regressor':
            errors = sqrt(mean_squared_error(test_labels, df))
            return errors
        else:
            return 'No RMSE for Classification'

    def load(self, path, type):
        self.model = load_model(path)
        self.type = type
        return self.model
