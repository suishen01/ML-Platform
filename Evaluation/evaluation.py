import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score


class Evaluation:

    evaluation = None
    test_labels = None

    def __init__(self, evaluation, test_labels=None):
        self.evaluation = evaluation
        self.test_labels = test_labels

    def getAbsError(self):

        # Calculate the absolute errors
        errors = abs(self.evaluation - self.test_labels)
        # Print out the mean absolute error (mae)

        return errors

    def getRSquare(self, output='single'):
        if output == 'multiple':
            r2s = r2_score(self.test_labels, self.evaluation, multioutput='variance_weighted')
        else:
            r2s = r2_score(self.test_labels, self.evaluation)


        print('R2 score: ', r2s)

        return r2s

    def getMSE(self):
        errors = mean_squared_error(self.test_labels, self.evaluation)
        print('Mean Square Error:', errors)
        return errors

    def getMAPE(self):
        errors = np.mean(np.abs((self.test_labels - self.evaluation) / self.test_labels)) * 100
        print('Mean Absolute Percentage Error:', errors)
        return errors

    def getRMSE(self):
        errors = sqrt(mean_squared_error(self.test_labels, self.evaluation))
        print('Root Mean Square Errors:', errors)
        return errors

    def getAccuracy(self):
        errors = accuracy_score(self.test_labels, self.evaluation)
        print('Accuracy:', errors)
        return errors
