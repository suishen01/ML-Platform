from MachineLearningModels.model import Model
from sklearn.svm import SVC
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np

class KernelSVM(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None, kernel='poly', degree=8):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = 'classifier'
        self.model = SVC(kernel=kernel, degree=degree)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Kernel SVM Train started............')
        self.model.fit(self.X, self.Y)
        print('Kernel SVM completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def save(self, filename='kernelsvm_model.pkl'):
        pickle.dump(self.model, open(filename, 'wb'))

    def featureImportance(self):
#        if X_headers is None:
#            X_headers = list(self.X)


#        feature_importance_ = zip(self.model.coef_[0], X_headers)
#        feature_importance = set(feature_importance_)
        return 'No feature importance for kernel svm'

    def getAccuracy(self, test_labels, predictions):
        df = pd.DataFrame(data=predictions.flatten())
        correct = 0
        for i in range(len(df)):
            if (df.values[i] == test_labels.values[i]):
                correct = correct + 1
        return correct/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        df = pd.DataFrame(data=predictions.flatten())
        index = 0
        for label_header in label_headers:
            classes = test_labels[label_header].unique()
            title = 'Normalized confusion matrix for Ridge (' + label_header + ')'
            self.plot_confusion_matrix(test_labels.ix[:,index], df.ix[:,index], classes=classes, normalize=True,
                      title=title)
            index = index + 1

    def getRSquare(self, test_labels, predictions, mode='single'):
        return 'No RSquare for Classification'

    def getMSE(self, test_labels, predictions):
        return 'No MSE for Classification'

    def getMAPE(self, test_labels, predictions):
        return 'No MAPE for Classification'

    def getRMSE(self, test_labels, predictions):
        return 'No RMSE for Classification'
