from MachineLearningModels.model import Model
from sklearn.linear_model import Lasso as LassoRegression
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
import json

class AdaptiveLasso(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None



    def __init__(self, X=None, Y=None, label_headers=None,  alpha=1, n_itr=5, type='regressor', cfg=False):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type
        self.cfg = cfg
        self.alpha = alpha
        self.n_itr = n_itr
        self.mapping_dict = None
        self.label_headers = label_headers

        self.model = LassoRegression(alpha=alpha)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        if self.type == 'classifier':
            self.Y = self.map_str_to_number(self.Y)

        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

        n_samples, n_features = self.X.shape
        p_obj = lambda w: 1. / (2 * n_samples) * np.sum((self.Y - np.dot(self.X, w)) ** 2) \
                          + alpha * np.sum(g(w))

        weights = np.ones(n_features)

        X_w = self.X / weights[np.newaxis, :]

        adaptive_lasso = LassoRegression(alpha=self.alpha, fit_intercept=False)

        adaptive_lasso.fit(X_w, self.Y)
        n_lasso_iterations = self.n_itr
        print('Adaptive Lasso Train started............')
        for k in range(n_lasso_iterations):
            X_w = self.X / weights[np.newaxis, :]
            adaptive_lasso = LassoRegression(alpha=self.alpha, fit_intercept=False)
            adaptive_lasso.fit(X_w, self.Y)
            coef_ = adaptive_lasso.coef_ / weights
            weights = gprime(coef_)
        self.model = adaptive_lasso
        print('Adaptive Lasso completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        if self.type == 'classifier':
            predictions = predictions.round()
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        if self.cfg:
            f = open('adaptivelasso_configs.txt', 'w')
            f.write(json.dumps(self.model.get_params()))
            f.close()
        print('No models will be saved for Adaptive lasso')

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
                title = 'Normalized confusion matrix for Adaptive Lasso (' + label_header + ')'
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
