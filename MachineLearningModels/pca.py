from MachineLearningModels.model import Model
from sklearn.decomposition import PCA as PCAmodel
import pandas as pd

class PCA(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None
    n_components = 2


    def __init__(self, X=None, Y=None,  n_components=2):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.n_components = n_components
        self.model = PCAmodel(n_components=n_components)


    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PCA Train started............')
        self.model.fit(self.X, self.Y)
        print('PCA completed..........')

        return self.model

    def fit_transform(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('PCA Train/Transform started............')
        self.model.fit_transform(self.X, self.Y)
        print('PCA completed..........')

        return self.model

    def save(self):
        print('No models will be saved for PCA')

    def featureImportance(self, X_headers=None):
        if X_headers is None:
            X_headers = list(self.X)

        index = []
        for i in range(self.n_components):
            index.append('PC-' + str(i))

        return pd.DataFrame(self.model.components_,columns=X_headers,index = index)
