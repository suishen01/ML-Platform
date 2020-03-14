from MachineLearningModels.model import Model
from sklearn.linear_model import Ridge as RidgeRegression
from sklearn.linear_model import RidgeClassifier
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np

class Ridge(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None,  alpha=1, type='regressor'):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type

        if self.type == 'regressor':
            self.model = RidgeRegression(alpha=alpha)
        else:
            self.model = RidgeClassifier(alpha=alpha)



    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Ridge Regression Train started............')
        self.model.fit(self.X, self.Y)
        print('Ridge Regression completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions[:,0]

    def save(self, filename='ridge_model.pkl'):
        pickle.dump(self.model, open(filename, 'wb'))

    def featureImportance(self):
#        if X_headers is None:
#            X_headers = list(self.X)


#        feature_importance_ = zip(self.model.coef_[0], X_headers)
#        feature_importance = set(feature_importance_)
        return self.model.coef_[0]

    def getAccuracy(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'classifier':
            correct = 0
            for i in range(len(df)):
                if (df.values[i] == test_labels.values[i]):
                    correct = correct + 1
            return correct/len(df)
        else:
            return 'No Accuracy for Regression'

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        df = pd.DataFrame(data=predictions.flatten())
        if self.type == 'classifier':
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                title = 'Normalized confusion matrix for Ridge (' + label_header + ')'
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
