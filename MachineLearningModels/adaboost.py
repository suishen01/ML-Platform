from sklearn.ensemble import AdaBoostRegressor
from MachineLearningModels.model import Model

class AdaBoost(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None,  n_estimators=100):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = AdaBoostRegressor(n_estimators=n_estimators)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('AdaBoost Train started............')
        self.model.fit(self.X, self.Y)
        print('AdaBoost Train completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def featureImportance(self, X_headers=None):
        if X_headers is None:
            X_headers = list(self.X)

        # Get numerical feature importances
        importances = list(self.model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_headers, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances
        [print('Variable: {!s:20} Importance: {}'.format(*pair)) for pair in feature_importances];

        return feature_importances
