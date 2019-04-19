import numpy as np
import pandas as pd
from DataPreprocessor.dataSQL import SQL


chunksize = 10
for data in pd.read_csv("data_small_withLaggedOutputs.csv", chunksize=chunksize):
    data = data[['gvkey', 'date', 'ret']].copy()
    sql = SQL(data)
    #sql.setIndex(['gvkey','date'])
    print(sql.getKeyList('gvkey'))
    tmp = sql.select('gvkey', where=12994)
    #print(tmp)
    tmpsql = SQL(tmp)
    tmpsql.sort(['gvkey', 'date'])
    print(tmpsql.data)
    break
