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
from MachineLearningModels.gradientboost import GradientBoost
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

# AdaBoost
rfr_ada = AdaBoost(n_estimators=5, type='classifier')
print('Train started............')
rfr_ada.fit(train_features, train_labels)
print('Train completed..........')
print('Prediction started............')
predictions_ada = rfr_ada.predict(test_features)
print('Prediction completed..........')

df_ada = pd.DataFrame(data=predictions_ada.flatten())
df_ada = df_ada.ix[:,0].map({'M': 0, 'B': 1})
test_labels = test_labels.ix[:,0].map({'M': 0, 'B': 1})

fpr_ada, tpr_ada, _ = roc_curve(test_labels, df_ada)

#GradientBoost
rfr_gb = GradientBoost(n_estimators=5, type='classifier')
print('Train started............')
rfr_gb.fit(train_features, train_labels)
print('Train completed..........')
print('Prediction started............')
predictions_gb = rfr_gb.predict(test_features)
print('Prediction completed..........')

df_gb = pd.DataFrame(data=predictions_gb.flatten())
df_gb = df_gb.ix[:,0].map({'M': 0, 'B': 1})

fpr_gb, tpr_gb, _ = roc_curve(test_labels, df_gb)

#PCA
pca = PCA(n_components=2)
train_features = pca.fit_transform(train_features)

test_features = pca.fit_transform(test_features)
rfr_pca = Ridge(type='classifier')
print('Train started............')
rfr_pca.fit(train_features, train_labels)
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions_pca = rfr_pca.predict(test_features)
print('Prediction completed..........')

df_pca = pd.DataFrame(data=predictions_pca.flatten())
df_pca = df_pca.ix[:,0].map({'M': 0, 'B': 1})

fpr_pca, tpr_pca, _ = roc_curve(test_labels, df_pca)


#RandomForest
rfr_rf = RandomForest(train_features, train_labels, type='classifier')
print('Train started............')
rfr_rf.fit()
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions_rf = rfr_rf.predict(test_features)
print('Prediction completed..........')

df_rf = pd.DataFrame(data=predictions_rf.flatten())
df_rf = df_rf.ix[:,0].map({'M': 0, 'B': 1})

fpr_rf, tpr_rf, _ = roc_curve(test_labels, df_rf)


#Ridge
rfr_ridge = Ridge(type='classifier')
print('Train started............')
rfr_ridge.fit(train_features, train_labels)
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions_ridge = rfr_ridge.predict(test_features)
print('Prediction completed..........')

df_ridge = pd.DataFrame(data=predictions_ridge.flatten())
df_ridge = df_ridge.ix[:,0].map({'M': 0, 'B': 1})

fpr_ridge, tpr_ridge, _ = roc_curve(test_labels, df_ridge)

# NeuralNetwork, this should always be the last one because its using the different I/Os
data = csvreader.read('data.csv')
data = data.drop(columns='Unnamed: 32')
data.set_index('id', inplace=True)

data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

labels = data[['diagnosis']].copy()

features = data.drop(columns='diagnosis')
feature_list = list(features)

train_features, test_features = np.split(features, [int(.9*len(features))])
train_labels, test_labels = np.split(labels, [int(.9*len(labels))])

model = load_model('models/breast_nn.h5')
predictions=model.predict(test_features)
fpr_nn, tpr_nn, _ = roc_curve(test_labels, predictions)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_ada, tpr_ada, label='AdaBoost')
plt.plot(fpr_gb, tpr_gb, label='GradientBoost')
plt.plot(fpr_nn, tpr_nn, label='NeuralNetwork')
plt.plot(fpr_pca, tpr_pca, label='PCA')
plt.plot(fpr_rf, tpr_rf, label='RandomForest')
plt.plot(fpr_ridge, tpr_ridge, label='Ridge')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#plt.figure(2)
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_ada, tpr_ada, label='AdaBoost')
#plt.plot(fpr_gb, tpr_gb, label='GradientBoost')
#plt.plot(fpr_nn, tpr_nn, label='NeuralNetwork')
#plt.plot(fpr_pca, tpr_pca, label='PCA')
#plt.plot(fpr_rf, tpr_rf, label='RandomForest')
#plt.plot(fpr_ridge, tpr_ridge, label='Ridge')
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve (zoomed in at top left)')
#plt.legend(loc='best')
#plt.show()
