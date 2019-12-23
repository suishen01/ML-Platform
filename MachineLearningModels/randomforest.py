from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from MachineLearningModels.model import Model
import pandas as pd
import pickle

class RandomForest(Model):

    # X represents the features, Y represents the labels
    X = None
    Y = None
    prediction = None
    model = None

    def __init__(self):
        pass

    def __init__(self, X=None, Y=None,  n_estimators=100, type='regressor'):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        self.type = type

        if (type == 'regressor'):
            self.model = RandomForestRegressor(n_estimators=n_estimators, verbose=0)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators, verbose=0)

    def fit(self, X=None, Y=None):
        if X is not None:
            self.X = X

        if Y is not None:
            self.Y = Y

        print('Random Forest Train started............')
        self.model.fit(self.X, self.Y)
        print('Random Forest Train completed..........')

        return self.model

    def predict(self, test_features):
        print('Prediction started............')
        self.predictions = self.model.predict(test_features)
        print('Prediction completed..........')
        return self.predictions

    def save(self, filename='randomforest_model.pkl'):
        pickle.dump(self.model, open(filename, 'wb'))

    def featureImportance(self):

        # Get numerical feature importances
        #importances = list(self.model.feature_importances_)
        # List of tuples with variable and importance
        #feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_headers, importances)]
        # Sort the feature importances by most important first
        #feature_importances = sorted(self.model.feature_importances_, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances
        #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

        return self.model.feature_importances_

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
                title = 'Normalized confusion matrix for RandomForest (' + label_header + ')'
                self.plot_confusion_matrix(test_labels.ix[:,index], df.ix[:,index], classes=classes, normalize=True,
                          title=title)
                index = index + 1
        else:
            return 'No Confusion Matrix for Regression'
