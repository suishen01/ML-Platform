import numpy as np
from sklearn.metrics import r2_score

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
        print('Mean Absolute Error:', round(np.mean(errors), 2),'degrees.')

        return errors

    def getRSquare(self, output='single'):
        if output == 'multiple':
            r2s = r2_score(self.test_labels, self.evaluation, multioutput='variance_weighted')

        print('R2 score: ', r2s)

        return r2s
