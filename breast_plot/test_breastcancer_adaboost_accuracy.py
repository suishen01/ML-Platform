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

csvreader = CsvReader()
data = csvreader.read('data.csv')
data = data.drop(columns='Unnamed: 32')
data.set_index('id', inplace=True)

#data = data[data.gvkey != 12994]
#cleaner = Cleaner(data)
#data = cleaner.clean(strategy='ignore')


labels = data[['diagnosis']].copy()

features = data.drop(columns='diagnosis')
feature_list = list(features)

#csvreader = CsvReader()
#data = csvreader.read('data_small_withLaggedOutputs.csv')
#labels = data[['ret','retrf','maxret']].copy()
#features = data[['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']].copy()

train_features, test_features = np.split(features, [int(.9*len(features))])
train_labels, test_labels = np.split(labels, [int(.9*len(labels))])

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Perform Grid-Search
#gsc = GridSearchCV(
#    estimator=RandomForestRegressor(),
#    param_grid={
#       'max_depth': range(9,10),
#       'n_estimators': (100, 1000),
#    },
#    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
#
#best_params = grid_result.best_params_
#grid_result = gsc.fit(train_features, train_labels)
#
rfr = AdaBoost(n_estimators=5, type='classifier')
print('Train started............')
rfr.fit(train_features, train_labels)
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions = rfr.predict(test_features)
print('Prediction completed..........')

correct = 0

df = pd.DataFrame(data=predictions.flatten())
for i in range(len(df)):
    if (df.values[i] == test_labels.values[i]):
        correct = correct + 1
print(correct/len(df))
