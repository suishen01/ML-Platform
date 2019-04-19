# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.dataSQL import SQL
from DataPreprocessor.datacleaner import Cleaner

chunksize = 500
for data in pd.read_csv("data_small_withLaggedOutputs.csv", chunksize=chunksize):
    sql = SQL(data)
    tmp = sql.select('gvkey', where=12994)
    tmpsql = SQL(tmp)
    tmpsql.sort(['date'])
    break

features = tmpsql.data[['dp','epm','bmm','ntis','tbl','tms','dfy','svar','bm','ep','maxret','absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','gma','grCAPX','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade','ret_lagged','retrf_lagged','maxret_lagged']].copy()
labels = tmpsql.data[['ret']].copy()

cleaner1 = Cleaner(labels)
cleaner2 = Cleaner(features)
labels = cleaner1.clean()
features = cleaner2.clean()

pkl_filename = 'models/random_forest_model.pkl'

model = RandomForest()
model.load(pkl_filename)

predictions = model.predict(features)

predicted_ret = []

for p in predictions:
    predicted_ret.append(p[0])

date_list = tmpsql.data[['date']].copy()

date_array = []

for d in date_list.values:
    date_array.append(datetime.datetime.strptime(str(d[0]), '%Y%m%d'))

predictions_data = pd.DataFrame(data = {'date': date_array, 'prediction': predicted_ret})

# Plot the actual values
plt.plot(predictions_data['date'], cleaner1.data, 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'r-', label = 'prediction')
plt.xticks(rotation = '60');
plt.legend()
# Graph labels
plt.xlabel('Date')
plt.ylabel('Ret')
plt.title('Actual and Predicted Values')
plt.show()
