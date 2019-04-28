from MachineLearningModels.randomforest import RandomForest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.csvread import CsvReader
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.datacleaner import Cleaner
from DataPreprocessor.dataspliter import Spliter
from Evaluation.evaluation import Evaluation
import seaborn as sns


pkl_filename = 'models/random_forest_model.pkl'

model = RandomForest()
model.load(pkl_filename)

feature_list =['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']

feature_importance = model.featureImportance(X_headers=feature_list)

feat_importances = pd.Series(feature_importance, index=feature_list)
toplot = feat_importances.nlargest(10).sort_values()
#toplot.plot(kind='barh')
#plt.show()


corr_list = np.append(feat_importances.nlargest(10).index.values, ['ret'])
print(corr_list)

csvreader = CsvReader()
data = csvreader.read('data_small_withLaggedOutputs.csv')

labels = data[['ret','retrf','maxret']].copy()
features = data[corr_list].copy()

cleaner1 = Cleaner(labels)
cleaner2 = Cleaner(features)
labels = cleaner1.clean()
features = cleaner2.clean()

corrmat = cleaner2.data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn",linewidths=.5)
plt.show()
