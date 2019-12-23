from MachineLearningModels.model import Model
from sklearn.linear_model import Ridge as RidgeRegression
from sklearn.linear_model import RidgeClassifier
import pandas as pd
import pickle

class Ridge(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None,  alpha=1, type='regressor'):

        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type

        if self.type == 'regressor':
            self.model = RidgeRegression(alpha=alpha)
        else:
            self.model = RidgeClassifier(alpha=alpha)



    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Ridge Regression Train started............')
        self.model.fit(self.X, self.Y)
        print('Ridge Regression completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def save(self, filename='ridge_model.pkl'):
        pickle.dump(self.model, open(filename, 'wb'))

    def featureImportance(self):
#        if X_headers is None:
#            X_headers = list(self.X)


#        feature_importance_ = zip(self.model.coef_[0], X_headers)
#        feature_importance = set(feature_importance_)
        return self.model.coef_[0]

    def getAccuracy(self, test_labels, predictions):
        correct = 0
        df = pd.DataFrame(data=predictions.flatten())
        for i in range(len(df)):
            if (df.values[i] == test_labels.values[i]):
                correct = correct + 1
        return correct/len(df)

    def getConfusionMatrix(self, test_labels, predictions, label_headers):
        if self.type == 'classifier':
            df = pd.DataFrame(data=predictions.flatten())
            index = 0
            for label_header in label_headers:
                classes = test_labels[label_header].unique()
                title = 'Normalized confusion matrix for Ridge (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df.ix[:,index], classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
            return 'No Confusion Matrix for Regression'
