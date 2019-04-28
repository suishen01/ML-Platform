# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from MachineLearningModels.randomforest import RandomForest
from DataPreprocessor.dataSQL import SQL
from DataPreprocessor.datacleaner import Cleaner
import csv

chunksize = 700
for data in pd.read_csv("data_small_withLaggedOutputs.csv", chunksize=chunksize):
    sql = SQL(data)
    tmp = sql.select('gvkey', where=12994)
    tmp2 = sql.select('gvkey', where=19049)
    tmp = tmp.append(tmp2, ignore_index=True)
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

#print(predictions)

predicted_ret = []

for p in predictions:
    predicted_ret.append(p[0])

date_list = tmpsql.data[['date']].copy()

date_array = []

for d in date_list.values:
    date_array.append(datetime.datetime.strptime(str(d[0]), '%Y%m%d'))

predictions_data = pd.DataFrame(data = {'date': date_array, 'prediction': predicted_ret})

with open('test_data_for_app.csv', mode='w') as training_data:
    training_data = csv.writer(training_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    print(tmpsql.data.shape)
    training_data.writerow(['index', 'gvkey', 'date', 'ret', 'predicted_ret'])
    for i in range(tmpsql.data.shape[0]):
        training_data.writerow([i, tmpsql.data[['gvkey']].iloc[[i]].values[0,0], date_array[i], tmpsql.data[['ret']].iloc[[i]].values[0,0], predicted_ret[i]])


#    i in range(1000):
#        for j in range(1000):
#            for k in range(20):
#                Df = (i-500)/1000
#                Dp = (j-500)/1000
#                DpeExternal = (k-10)/10
#                Dpref = ((Kp / R + 1) * Df + Kp * (Dp + DpeExternal) + (-Dp / (2 * Pi * T0) - Df) * Tp) / Kp
#                training_data.writerow([Df, Dp, DpeExternal, Dpref])
#        print(i)
