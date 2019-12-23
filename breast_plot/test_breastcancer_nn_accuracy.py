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
from MachineLearningModels.gradientboost import GradientBoost
from MachineLearningModels.linearregression import LinearRegression
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from eli5.permutation_importance import get_score_importances
import numpy as np

csvreader = CsvReader()
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


correct = 0

df = pd.DataFrame(data=predictions.flatten())
for i in range(len(df)):
    if df.values[i] >= 0.5:
        df.values[i] = 1
    else:
        df.values[i] = 0
    if (df.values[i] == test_labels.values[i]):
        correct = correct + 1
print(correct/len(df))
