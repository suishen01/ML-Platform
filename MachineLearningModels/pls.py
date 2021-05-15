from MachineLearningModels.model import Model
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
import json

class PLS(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None,  n_components=2, type='regressor', cfg=False):
        self.name = 'PLS'

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type
        self.cfg = cfg
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PLS Train started............')
        self.model.fit(self.X, self.Y)
        print('PLS completed..........')

        return self.model


    def fit_transform(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PLS Train/Transform started............')
        self.X  = self.model.fit_transform(self.X)
        print('PLS completed..........')

        self.X = pd.DataFrame(self.X)
        return self.X

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        if self.cfg:
            f = open('pls_configs.txt', 'w')
            f.write(json.dumps(self.model.get_params()))
            f.close()
        print('No models will be saved for PLS')

    def featureImportance(self):
    #    if X_headers is None:
    #        X_headers = list(self.X)
#
#        feature_importance_ = zip(self.model.coef_.reshape(1,-1)[0], X_headers)
#        feature_importance = set(feature_importance_)

        return self.model.coef_


    def getAccuracy(self, test_labels, predictions, origin=0, hitmissr=0.8):
        correct = 0
        df = pd.DataFrame(data=predictions.flatten())
        for i in range(len(df)):
            if 1 - abs(df.values[i] - test_labels.values[i])/abs(df.values[i]) >= hitmissr:
                correct = correct + 1
        return float(correct)/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        return 'No Confusion Matrix for Regression'

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
