from MachineLearningModels.model import Model
from sklearn.linear_model import LinearRegression as LinearRegressionModel

class LinearRegression(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = LinearRegressionModel()


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Linear Regression Train started............')
        self.model.fit(self.X, self.Y)
        print('Linear Regression completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        print('No models will be saved for lasso')

    def featureImportance(self):
        #if X_headers is None:
        #    X_headers = list(self.X)
        #print(self.model.coef_)
        #feature_importance_ = zip(self.model.coef_[0], X_headers)
        #feature_importance = set(feature_importance_)

        return self.model.coef_
