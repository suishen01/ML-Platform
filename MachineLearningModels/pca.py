from MachineLearningModels.model import Model
from sklearn.decomposition import PCA as PCAmodel
import pandas as pd
from MachineLearningModels.ridge import Ridge
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np

class PCA(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    n_components = 2
    ridge_model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None,  n_components=2, type='regressor'):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type
        self.n_components = n_components
        self.model = PCAmodel(n_components=n_components)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PCA Train started............')
        self.X = self.model.fit_transform(self.X)
        print('PCA completed..........')

        self.ridge_model = Ridge(type=self.type)
        print('Ridge Train started............')
        self.ridge_model.fit(self.X, self.Y)
        print('Ridge completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        test_features = self.model.fit_transform(test_features)
        self.predictions = self.ridge_model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def save(self, filename='pcaridge_model.pkl'):
        pickle.dump(self.ridge_model, open(filename, 'wb'))

    def featureImportance(self):
        index = []
        #for i in range(self.n_components):
        #    index.append('PC-' + str(i))
        index.append('PC-' + str(0))
        # return pd.DataFrame(self.model.components_[0].reshape((1,30)),columns=X_headers,index = index)
        return self.model.components_[0]


    def getAccuracy(self, test_labels, predictions):
        correct = 0
        df = pd.DataFrame(data=predictions.flatten())
        for i in range(len(df)):
            if (df.values[i] == test_labels.values[i]):
                correct = correct + 1
        return correct/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        if self.type == 'classifier':
            df = pd.DataFrame(data=predictions.flatten())
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                title = 'Normalized confusion matrix for PCA+Ridge (' + label_header + ')'
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
