import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from MachineLearningModels.lasso import Lasso
from Utils.csvread import CsvReader
from MachineLearningModels.adaboost import AdaBoost
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation

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

alpha = 0.001
lasso = Lasso()

lasso.fit(train_features, train_labels)
predictions = lasso.predict(test_features)
predict = Evaluation(evaluation=predictions, test_labels=test_labels)
r2s = predict.getRSquare(output='multiple')
print('R2 score: ', r2s)
