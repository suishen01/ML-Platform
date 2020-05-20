from MachineLearningModels.model import Model
from sklearn.linear_model import ElasticNet as ElasticNetModel

class ElasticNet(Model):

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

        self.model = ElasticNetModel()


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('ElasticNet Train started............')
        self.model.fit(self.X, self.Y)
        print('ElasticNet completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions


    def save(self):
        print('No models will be saved for lasso')

    def featureImportance(self):

        return self.model.coef_
