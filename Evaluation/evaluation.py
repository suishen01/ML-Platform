import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score


class Evaluation:

    output_dir = None
    evaluation = None
    test_labels = None

    def __init__(self, evaluation, test_labels=None, output_dir='./results.txt'):
        self.evaluation = evaluation
        self.test_labels = test_labels
        self.output_dir = output_dir

    def getAbsError(self):

        # Calculate the absolute errors
        errors = abs(self.evaluation - self.test_labels)
        # Print out the mean absolute error (mae)
        text_file = open(self.output_dir, "a")
        text_file.write("Absolute Error: %s" % str(errors))
        text_file.close()
        return errors

    def getRSquare(self, output='single'):
        if output == 'multiple':
            errors = r2_score(self.test_labels, self.evaluation, multioutput='variance_weighted')
        else:
            errors = r2_score(self.test_labels, self.evaluation)

        text_file = open(self.output_dir, "a")
        text_file.write("R Square Score: %s" % str(errors))
        text_file.close()
        return errors

    def getMSE(self):
        errors = mean_squared_error(self.test_labels, self.evaluation)

        text_file = open(self.output_dir, "a")
        text_file.write("Mean Squared Error: %s" % str(errors))
        text_file.close()
        return errors

    def getMAPE(self):
        errors = np.mean(np.abs((self.test_labels - self.evaluation) / self.test_labels)) * 100

        text_file = open(self.output_dir, "a")
        text_file.write("Mean Absolute Percenatge Error: %s" % str(errors))
        text_file.close()

        return errors

    def getRMSE(self):
        errors = sqrt(mean_squared_error(self.test_labels, self.evaluation))

        text_file = open(self.output_dir, "a")
        text_file.write("Squared Mean Squared Error: %s" % str(errors))
        text_file.close()
        return errors

    def getAccuracy(self):
        errors = accuracy_score(self.test_labels, self.evaluation)
        text_file = open(self.output_dir, "a")
        text_file.write("Accuracy: %s" % str(errors))
        text_file.close()
        return errors
