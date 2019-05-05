from MachineLearningModels.model import Model
from sklearn.cross_decomposition import PLSRegression

class PLS(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None,  n_components=2):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = PLSRegression(n_components=n_components)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PLS Train started............')
        self.model.fit(self.X, self.Y)
        print('PLS completed..........')

        return self.model

    def fit_transform(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PLS Train/Transform started............')
        self.model.fit_transform(self.X, self.Y)
        print('PLS completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        print('No models will be saved for PLS')

    def featureImportance(self, X_headers=None):
        if X_headers is None:
            X_headers = list(self.X)

        feature_importance_ = zip(self.model.coef_.reshape(1,-1)[0], X_headers)
        feature_importance = set(feature_importance_)

        return feature_importance
