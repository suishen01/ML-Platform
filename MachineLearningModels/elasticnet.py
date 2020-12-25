from MachineLearningModels.model import Model
from sklearn.linear_model import ElasticNet as ElasticNetModel
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np

class ElasticNet(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None, label_headers=None,  l1_ratio=1, type='regressor'):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type

        self.mapping_dict = None
        self.label_headers = label_headers

        self.model = ElasticNetModel(l1_ratio=l1_ratio)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        if self.type == 'classifier':
            self.Y = self.map_str_to_number(self.Y)
            
        print('ElasticNet Train started............')
        self.model.fit(self.X, self.Y)
        print('ElasticNet completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        if self.type == 'classifier':
            predictions = predictions.round()
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        print('No models will be saved for elasticnet')

    def featureImportance(self):

        return self.model.coef_

    def map_str_to_number(self, Y):
    mapping_flag = False
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
