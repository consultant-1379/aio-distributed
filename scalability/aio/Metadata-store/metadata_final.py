#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
import time
import os
# for trend detection
import sys
sys.path.append("..")

from configs import bad_direction_kpi_dict as dir_dict 
from configs import kpi_constraints_dict
from configs.tests2 import drop_missing_data, test_stationarity, test_seasonality, ACF_condition
from configs.tests2 import test_seasonality_vectorized, pivot_maker_vectorized, seas_testing_vectorized

from collections import OrderedDict
# kpi_constraints_dict for is_it_constant

from statsmodels.tsa.stattools import acf
from datetime import datetime
import datetime

#pd.set_option('display.max_rows', None)
import dask.dataframe as dd


# In[2]:


import dask
from dask.distributed import Client
client = Client( dashboard_address = ':44594', n_workers = 68, threads_per_worker = 2 ) #scheduler_address=':37243'
client


# In[3]:


# Data related stuff #
def regex_file_filter(s, patt='4weeks-\S*\.csv'):
    pattern = re.compile(patt)
    return pattern.match(s) != None


def clean(df, col_id): # dropping useless rows and columns
    df_clean = df.copy()
    df_clean = df_clean[df_clean[col_id].notnull()] # useful rows

    df_desc = df_clean.describe(datetime_is_numeric=True).T
    cols_to_use = list(df_desc[(df_desc['std']!=0.0)&(df_desc['count']>1.0)].index) # useful cols
    
    if (col_id in cols_to_use): return df_clean[cols_to_use]
    return df_clean[[col_id, *cols_to_use]]


def data_step(string_file, col_id, sep=',', index_col=0):
    df = pd.read_csv(string_file, sep=sep, index_col=index_col)
    df_clean = clean(df, col_id)

    df_clean['dt'] = pd.to_datetime(df_clean.ts, unit='s') # gonna be the index
    return df_clean.sort_values(by='ts')


trim = lambda ser: ser.loc[ser.first_valid_index():ser.last_valid_index()]


def rowIndex(row):
    return row.name


def colIndex(row, new_end):
    col_idx, = np.where(np.array(row.index) == pd.Timestamp(new_end))
    return col_idx[0]


# In[4]:


file = "../data/4weeks-plan_name-1614153600.csv"
ts = "1614556800"
max_missing_ratio = .35
results = pd.DataFrame(columns=['ts',
                                'path',
                                'dimension_name',
                                'kpi_name',
                                'missing_data_ratio_all', 
                                'missing_data_ratio_last_week',
                                'seasonality_flag',
                                'statonarity_flag',
                                'missing_data_imputation_flag',
                                'table',
                                "nan_trimming_flag",
                                #'ACF_max_difference',     
                                'model_type'])

ts = int(file[-14:-4])
day_of_week = pd.to_datetime(ts,unit='s').dayofweek
ts = ts + ((3600 * 24) * (6-day_of_week))
ts = ts + ((24*3600) - ts % (24*3600))
end = pd.to_datetime(ts, unit='s')
last_week_start = end-pd.Timedelta('1 w')
start = end-pd.Timedelta('4 w')

col_id = "plan_name" #! debug, ezt elvileg a címből kéne kiolvasni
df = data_step( file , col_id)
labels = df[col_id].unique()


# In[6]:


def by_label(label):
    df_label = clean(df[df[col_id]==label], col_id=col_id)

    if (df_label.shape[0]<3): return None
    
    # reindexing, beacause timestamp may be missing
    time_slice = pd.date_range(df_label.dt.min(), df_label.dt.max(), freq='H')

    df_label = df_label.set_index('dt')
    df_label = df_label[~df_label.index.duplicated()].reindex(time_slice)
    df_list = [] #!debug
    for col in df_label.columns.drop([col_id,"ts"]): #* ~ 10
        if col not in dir_dict.keys():
            continue
        df2 = dask.delayed(one_kpi)(col)
        df_list.append(df2) #!debug
    return df_list #!debug


# In[7]:


def one_kpi(col):
    df2 = df[[col_id, col, "dt"]]
    #df2 = df2.drop_duplicates(subset=[col_id, "dt"], keep="last")
    df2 = df2.sort_values([col_id,"dt"])
    df2 = df2.drop_duplicates(subset=[col_id, "dt"], keep="last")
    df2 = df2.pivot(index=col_id, columns="dt", values = col)
    df2["missing_data_ratio_all"] = df2.apply(lambda x: 1-sum(~x.isna())/(4*7*24), axis=1)
    df2["missing_data_ratio_last_week"] = df2.apply(lambda x: 1-sum(~x[last_week_start:].isna())/(7*24),axis=1)
    df2["path"] = file
    df2["ts"] = ts
    df2["table"] = col_id + str(ts)
    df2["dimension_name"] = df2.apply(lambda x: OrderedDict([(col_id, x.name)]), axis=1)
    #df2["kpi_name"] = col
    df2["model_type"] = df2.apply(lambda x: "bad_quality_data" if x["missing_data_ratio_all"] > max_missing_ratio else None, axis=1)
    upper, lower = kpi_constraints_dict[col]
    df2["model_type"] = df2.apply(lambda x: "constant" if ((x[:-6].quantile(0.51)==lower) or (x[:-6].quantile(0.49)==upper))&(x.model_type!="bad_quality_data") else x.model_type, axis=1)

    #model_type_lambda2 = lambda x: "constant" if ((x[:-6].quantile(0.51)==lower) or (x[:-6].quantile(0.49)==upper))&(x.model_type!="bad_quality_data") else x.model_type
    #df2["model_type"] = model_type_lambda2(df2)

    df_bad_quality = df2[(df2["model_type"]=="bad_quality_data")|(df2["model_type"]=="constant")]

    df_bad_quality[['seasonality_flag','statonarity_flag','missing_data_imputation_flag', "nan_trimming_flag"]] = None,None,None,None

    df2 = df2[df2["model_type"].isna()]
    if df2.shape[0]>0:
        df2[["nan_trimming_flag", "new_start", "new_end"]] = df2.apply(lambda x: drop_missing_data(np.array(x[:-7].dropna(axis=0).index, dtype='datetime64[h]'), 4),axis=1, result_type='expand')
        
        df2["missing_data_drop_flag"] = df2["nan_trimming_flag"]
        df2["start_index"] = df2.apply(lambda x: df2.columns.get_loc(pd.Timestamp(x["new_start"])), axis=1)
        df2["end_index"] = df2.apply(lambda x: df2.columns.get_loc(pd.Timestamp(x["new_end"])), axis=1)
        
        
        df_change = df2[df2.nan_trimming_flag==True]
        df2 = df2[df2.nan_trimming_flag!=True]
#* innen folytassuk
        df_change[["missing_data_imputation_flag", "missing_data_drop_flag", "seasonality_flag"]] = df_change.apply(lambda x: (True, 1, test_seasonality_vectorized((pd.Series(x[x["start_index"]:x["end_index"]]).interpolate(method='pchip')))) 
                                                                            if sum(x[x["start_index"]:x["end_index"]].isna())!=0 
                                                                            else (False, 1, test_seasonality_vectorized(x[x["start_index"]:x["end_index"]])),
                                                                            axis=1, result_type='expand')

        # Ez lehet x[0:667]-re kene hogy fusson
        if df2.shape[0]>0:
            #df2[["missing_data_imputation_flag", "missing_data_drop_flag", "is_seas"]] = df2.apply(lambda x: (True, 0, test_seasonality_vectorized((pd.Series(x[x["start_index"]:x["end_index"]]).interpolate(method='pchip')))) if sum(x[x["start_index"]:x["end_index"]].isna())!=0 else (False, 0, test_seasonality_vectorized(x[x["start_index"]:x["end_index"]])),axis=1, result_type='expand')
            df2[["missing_data_imputation_flag", "missing_data_drop_flag", "seasonality_flag"]] = df2.apply(lambda x: (True, 0, test_seasonality_vectorized((pd.Series(x[0:667]).interpolate(method='pchip')))) if sum(x[0:667].isna())!=0 else (False, 0, test_seasonality_vectorized(x[0:667])),axis=1, result_type='expand')
        
        df2 = pd.concat([df2, df_change])

        df_change = None

        df_trend = df2[df2.seasonality_flag==False]
        df2 = df2[df2.seasonality_flag!=False]
        if df_trend.shape[0]>0:
            df_trend["statonarity_flag"] = df_trend.apply(lambda x: test_stationarity(x[0:667].dropna(),dir_dict[col]) ,axis=1)
        else:
            df2["statonarity_flag"] = None
        df2 = pd.concat([df2, df_trend])

        df_trend = None
        df2.loc[(df2['seasonality_flag'] == True), 'model_type'] = 'seasonal_prophet'
        df2.loc[(df2['seasonality_flag'] != True) & (df2['statonarity_flag'] == True), 'model_type'] = 'non_seasonal_dbscan'
        df2.loc[(df2['seasonality_flag'] != True) & (df2['statonarity_flag'] != True), 'model_type'] = 'non_seasonal_trend'
    return df2


# In[8]:


tasks = [ by_label(label) for label in labels[:3] ]


# In[9]:


results = dask.compute(tasks)[0]


# In[13]:


col_num = 0
for label in labels[:3]:
    df_label = clean(df[df[col_id]==label], col_id=col_id)

    if (df_label.shape[0]<3): continue

    # reindexing, beacause timestamp may be missing
    time_slice = pd.date_range(df_label.dt.min(), df_label.dt.max(), freq='H')

    df_label = df_label.set_index('dt')
    df_label = df_label[~df_label.index.duplicated()].reindex(time_slice)
    df_list = [] 
    for col in df_label.columns.drop([col_id,"ts"]): #* ~ 10
        col_num +=1


# In[14]:


col_num # 83 timeseries

