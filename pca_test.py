import numpy as np
from MachineLearningModels.pca import PCA
from Utils.csvread import CsvReader
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import pandas as pd

csvreader = CsvReader()
data = csvreader.read('data_small_withLaggedOutputs.csv')
#for i in list(data):
#    print(i)
data.set_index(['gvkey', 'date'],inplace=True)

labels = data[['ret']].copy()
features = data[['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']].copy()

cleaner1 = Cleaner(labels)
cleaner2 = Cleaner(features)
labels = cleaner1.clean(type='df')
#labels = cleaner1.data
features = cleaner2.clean(type='df')
#features = cleaner2.data
#print(features)

spliter = Spliter()
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = spliter.split(features, labels, test_size = 0.25, random_state = 42)

#print(train_features)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Define pca object
pca = PCA()

# Fit
X_new = pca.fit_transform(train_features)
#print(X_new)
#X_array = np.array(X_new)
#print(X_array)
#print(X_new.shape)
#print(pca.featureImportance())
principalDf = pd.DataFrame(data = X_new, columns = ['PC-1', 'PC-2'])
principalDf.index = train_features.index
#train_features.reset_index(drop=True, inplace=True)
#train_labels.reset_index(drop=True, inplace=True)
#principalDf.reset_index(drop=True, inplace=True)

finalDf = pd.concat([principalDf, train_labels[['ret']]], axis = 1)
print(finalDf)
