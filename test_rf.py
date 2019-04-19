from Utils.csvread import CsvReader
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation


csvreader = CsvReader()
data = csvreader.read('data_small_withLaggedOutputs.csv')

data = data[data.gvkey != 12994]

labels = data[['ret','retrf','maxret']].copy()
features = data[['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']].copy()

cleaner1 = Cleaner(labels)
cleaner2 = Cleaner(features)
labels = cleaner1.clean()
features = cleaner2.clean()

spliter = Spliter()
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = spliter.split(features, labels, test_size = 0.25, random_state = 42)

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
rfr = RandomForest(train_features, train_labels)
print('Train started............')
rfr.fit()
print('Train completed..........')
# Use the forest's predict method on the test data
print('Prediction started............')
predictions = rfr.predict(test_features)
print('Prediction completed..........')

predict = Evaluation(evaluation=predictions, test_labels=test_labels)
r2s = predict.getRSquare(output='multiple')
print('R2 score: ', r2s)

rfr.save()
