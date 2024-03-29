from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from MachineLearningModels.model import Model
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
import json

class RandomForest(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None, label_headers=None,  n_estimators=100, type='regressor', cfg=False):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type
        self.cfg = cfg

        self.mapping_dict = None
        self.label_headers = label_headers

        if (type == 'regressor'):
            self.model = RandomForestRegressor(n_estimators=n_estimators, verbose=0)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators, verbose=0)

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        if self.type == 'classifier':
            self.map_str_to_number(Y)

        print('Random Forest Train started............')
        self.model.fit(self.X, self.Y)
        print('Random Forest Train completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def save(self, filename='randomforest_model.pkl'):
        if self.cfg:
            f = open('randomforest_configs.txt', 'w')
            f.write(json.dumps(self.model.get_params()))
            f.close()
        pickle.dump(self.model, open(filename, 'wb'))

    def featureImportance(self):

        # Get numerical feature importances
        #importances = list(self.model.feature_importances_)
        # List of tuples with variable and importance
        #feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_headers, importances)]
        # Sort the feature importances by most important first
        #feature_importances = sorted(self.model.feature_importances_, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances
        #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

        return self.model.feature_importances_

    def getAccuracy(self, test_labels, predictions, origin=0, hitmissr=0.8):
        if self.type == 'classifier':
            correct = 0
            df = pd.DataFrame(data=predictions.flatten())
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
        if self.type == 'classifier':
            df = pd.DataFrame(data=predictions.flatten())
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                title = 'Normalized confusion matrix for RandomForest (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df.ix[:,index], classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
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
