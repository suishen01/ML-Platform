from MachineLearningModels.model import Model
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
import json

class LogisticRegression(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None


    def __init__(self, X=None, Y=None, cfg=False):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.model = LogisticRegressionModel()
        self.cfg = cfg


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Logistic Regression Train started............')
        self.model.fit(self.X, self.Y)
        print('Logistic Regression completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        if self.cfg:
            f = open('logisticregression_configs.txt', 'w')
            f.write(json.dumps(self.model.get_params()))
            f.close()
        print('No models will be saved for Logistic Regression')

    def featureImportance(self):
        #if X_headers is None:
    #        X_headers = list(self.X)
    #    print(self.model.coef_)
    #    feature_importance_ = zip(self.model.coef_[0], X_headers)
    #    feature_importance = set(feature_importance_)

        return self.model.coef_
