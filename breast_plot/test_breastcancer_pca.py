from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from tensorflow.python.keras.models import load_model
from Utils.csvread import CsvReader
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from MachineLearningModels.ann import NeuralNetwork
from eli5.permutation_importance import get_score_importances
from MachineLearningModels.ridge import Ridge
from MachineLearningModels.lasso import Lasso
from MachineLearningModels.pls import PLS
from sklearn.metrics import mean_squared_error, r2_score
from MachineLearningModels.adaboost import AdaBoost
from MachineLearningModels.linearregression import LinearRegression
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from MachineLearningModels.pca import PCA

csvreader = CsvReader()
data = csvreader.read('data.csv')
data = data.drop(columns='Unnamed: 32')
data.set_index('id', inplace=True)

labels = data[['diagnosis']].copy()

features = data.drop(columns='diagnosis')
feature_list = list(features)

train_features, test_features = np.split(features, [int(.9*len(features))])
train_labels, test_labels = np.split(labels, [int(.9*len(labels))])

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Define pca object
pca = PCA(n_components=2)
train_features = pca.fit_transform(train_features)

test_features = pca.fit_transform(test_features)

rfr = Ridge(type='classifier')
print('Train started............')
rfr.fit(train_features, train_labels)
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions = rfr.predict(test_features)
print('Prediction completed..........')

predict = Evaluation(evaluation=predictions, test_labels=test_labels)
r2s = predict.getAccuracy()
print('Accuracy score: ', r2s)

feature_importance = pca.featureImportance()
feat_importance = pd.Series(feature_importance, index=feature_list)
toplot = feat_importance.nlargest(10).sort_values()
toplot.plot(kind='barh')
plt.show()
