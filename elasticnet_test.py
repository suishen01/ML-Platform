import numpy as np
from MachineLearningModels.elasticnet import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from Utils.csvread import CsvReader
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import pandas as pd

csvreader = CsvReader()
data = csvreader.read('CRSPMF_MFFFactors_FundSumM_Sample.csv')

#data = data[data.gvkey != 12994]

labels = data[['mret', 'Wmret', 'MFExRet', 'DollarFRet', 'DollarFWmRet']].copy()
labels = labels.drop(labels.index[[labels.shape[0]-1]])
features = data[['LogFunadAge','LogFundSize','retailFDummy','InstiFDummy','WFunddiv_ytd','WFundper_cash','WFundper_oth','WFundper_pref','WFundnav_52w_h','WFundnav_52w_l_dt','WFundper_com']].copy()
features = features.drop(features.index[[1]])
feature_list = ['LogFunadAge','LogFundSize','retailFDummy','InstiFDummy','WFunddiv_ytd','WFundper_cash','WFundper_oth','WFundper_pref','WFundnav_52w_h','WFundnav_52w_l_dt','WFundper_com']

cleaner1 = Cleaner(labels)
cleaner2 = Cleaner(features)
labels = cleaner1.clean()
labels = cleaner1.data
features = cleaner2.clean()
features = cleaner2.data

spliter = Spliter()
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = spliter.split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Define PLS object
ela = ElasticNet()

# Fit
ela.fit(train_features, train_labels)

# Prediction
Y_pred = ela.predict(test_features)

#print(pls.featureImportance())

# Calculate scores
score = r2_score(test_labels, Y_pred)
print(score)
mse = mean_squared_error(test_labels, Y_pred)
print(mse)

#X_new = pls.fit_transform(train_features, train_labels)
#X_array = np.array(X_new)
#print(X_array)
