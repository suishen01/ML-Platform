import numpy as np
from MachineLearningModels.pls import PLS
from sklearn.metrics import mean_squared_error, r2_score
from Utils.csvread import CsvReader
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation

csvreader = CsvReader()
data = csvreader.read('data_small_withLaggedOutputs.csv')

labels = data[['ret']].copy()
features = data[['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']].copy()

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
pls = PLS(n_components=5)

# Fit
pls.fit(train_features, train_labels)

# Prediction
Y_pred = pls.predict(test_features)

print(pls.featureImportance())

# Calculate scores
score = r2_score(test_labels, Y_pred)
print(score)
mse = mean_squared_error(test_labels, Y_pred)
print(mse)

X_new = pls.fit_transform(train_features, train_labels)
X_array = np.array(X_new)
print(X_array)
