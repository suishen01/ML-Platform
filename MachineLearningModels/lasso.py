from sklearn.ensemble import AdaBoostRegressor
from MachineLearningModels.model import Model
from sklearn.linear_model import Lasso as LassoRegression

class Lasso(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None,  alpha=1):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = LassoRegression(alpha=alpha)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Lasso Regression Train started............')
        self.model.fit(self.X, self.Y)
        print('Lasso Regression completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        print('No models will be saved for lasso')
