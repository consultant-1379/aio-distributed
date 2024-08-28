import prophet
import sys

sys.modules['fbprophet'] = prophet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import plotly.offline as pyo
import plotly.graph_objs as go

import itertools
import os, sys

from supersmoother import SuperSmoother, LinearSmoother
from sklearn.preprocessing import StandardScaler


from collections import OrderedDict
sys.path.append("..")
from configs.functions import make_future, add_cond_trend_version, run_prophet_funct, hparam_tuning
from configs.functions import preprocess_data
from configs.bad_direction_kpi_dict import bad_direction_kpi_dict
from configs.kpi_constraints_dict import kpi_constraints_dict

from uuid import uuid4

import dask
from dask.distributed import Client

import logging, sys


def main():

    client = Client( dashboard_address=':44594', n_workers = 5, threads_per_worker = 2) #scheduler_address=':37243'


    data_loc = "../../../data/" 
    
    file = "4weeks-subs_mcc-anon.csv"#"4weeks-subs_mcc-anon.csv"
    datas = pd.read_csv(data_loc+ file) #pd.read_csv( "../Data/" + file )
    print("-"*30,"DF READ","-"*30)

    metadata_store = pd.read_csv(data_loc + "metadata-anon.csv")#pd.read_csv('../Data/metadata_anon.csv')


    # Get rid of all irrelevant metadata
    metadata_store = metadata_store[ metadata_store.model_type == 'non_seasonal_trend' ]
    metadata_store = metadata_store[ metadata_store.path == file ]

    # evaluate str to OrderedDict
    metadata_store.dimension_name = metadata_store.dimension_name.map(str).map(lambda element: eval(element))

    #params
    missing_data_percentage_param = 0.3
    
    
    percent = 0.1 #used in myForecast
    scores = ['mae'] #['mdape', 'mape', 'smape', 'mae']
    predictions_write_to = ''
    errors_write_to = ''
    write_to = ''
    alpha = 1.0 #used in myForecast


    #infos
    end = pd.to_datetime(metadata_store['ts'].values[0], unit='s')
    ts = metadata_store['ts'].values[0]
    start = end - pd.Timedelta(4, unit = 'w') #used in myForecast
    files = metadata_store['path'].unique() # arr of unique files

    metadata_store["failed"] = [False] * len(metadata_store)
    metadata_store["num_fails"] = [0] * len(metadata_store)
    metadata_store["last_failed_ts"] = [None] * len(metadata_store)
    metadata_store["ts_uuid"] = metadata_store["failed"].map(lambda x : str(uuid4()))

    param_grid = {  'changepoint_prior_scale': [0.01, 0.1, 1.0],
                        'changepoint_range': [0.8, 0.9, 0.95]       }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    p_uuid_df = pd.DataFrame( { str(uuid4()): p  for p in all_params}).T   # p_uuid: 1 id per parameter combination  for tracking - same for all models with the same hparams, regardless of timeseries

    p_uuid_df.iloc[0].to_dict()
    for idx, row in p_uuid_df.head().iterrows():
        print(idx, row.to_dict())


    task_status = lambda : pd.Series([ [ fut.status for p_uuid, fut in compute[ts_uuid].items()] for ts_uuid in compute.keys() ]).explode().value_counts()
    failed_tasks = lambda : [[ (ts_uuid,p_uuid) for p_uuid, fut in compute[ts_uuid].items() if fut.status != "finished"] for ts_uuid in compute.keys() ]

    df_dict = {}
    for  _, mdrow in metadata_store.iterrows():
        df = preprocess_data( datas, mdrow, start, end )
        df_dict[ mdrow.ts_uuid ] = df

    taskgroups = {}

    for  _, mdrow in metadata_store.iterrows():
        df = df_dict[ mdrow.ts_uuid ]

        taskgroup = dask.delayed(submit_training)( df, mdrow, p_uuid_df)
        taskgroups[ mdrow.ts_uuid ] = ( taskgroup )

    compute = dask.compute( taskgroups )[0]
    time.sleep(20) # compute finishes before all submitted task, wait a little

    task_status() # possible task status: finished, pending, cancelled, lost, error, newly-created, closing, connecting, running, closed
    # finished is good, the others are bad statusesu

    zookeeper_table = pd.DataFrame(columns = ["ts_uuid", "p_uuid", "ts", "kpi", "dim", "params", "failed", "num_failed", "last_failed_ts"]) 
    # here ts in ts_uuid stands for timeseries, while ts is timestamp.
    # failed contains the status of the last retry
    zookeeper_table

    task_status()

    bad_status = ["pending", "cancelled", "lost", "error", "newly-created", "closing", "connecting", "running", "closed" ]

    for i in range(10):
        
        print(i)
        print(task_status())

        zookeeper_table = refresh_zk_table(zookeeper_table, compute, metadata_store,p_uuid_df)


        #check if there is an unfinished task
        if not any(status in task_status() for status in bad_status):
            break

        status_df = pd.DataFrame(compute).applymap(lambda x: x.status)
        unfinished_tasks = [(status_df.index[x], status_df.columns[y]) for x, y in zip(*np.where(status_df.values != 'finished'))]

        compute = retry_unfinished_tasks(compute,p_uuid_df,metadata_store,df_dict)

        
        
        time.sleep(0.5 * len(unfinished_tasks))

    status_df = pd.DataFrame(compute).applymap(lambda x: x.status)
    finished_tasks = [(status_df.index[x], status_df.columns[y]) for x, y in zip(*np.where(status_df.values == 'finished'))]  # tuple of (p_uuid, ts_uuid)

    results = {}

    for p_uuid, ts_uuid in finished_tasks: 
        
        if ts_uuid not in results:
            results[ts_uuid] = {}

        results[ts_uuid][p_uuid] = compute[ts_uuid][p_uuid].result()

    for ts, taskgroup in results.items():
        for p_uuid, task in taskgroup.items():
            params = p_uuid_df.loc[ p_uuid ]

            
            results[ts][p_uuid] = pd.concat([task.T, params]).T

    concated_results  = { ts_uuid : pd.concat([ v for k,v in results[ ts_uuid ].items() ]).reset_index(drop=True) for ts_uuid in results }


    detrended_timeseries = {}
    for _,mdrow in metadata_store.iterrows():
        df = df_dict[ mdrow.ts_uuid ]
        tuning_results = concated_results[ mdrow.ts_uuid ]

        past_ts = dask.delayed(myForecast)(df, tuning_results, start, end, mdrow, "mae")
        detrended_timeseries[mdrow.ts_uuid] = past_ts

    detrended_timeseries = dask.compute(detrended_timeseries)[0]


def submit_training(df, row, p_uuid_df):
    """
    Submit every model for training in dask.

    df - timeseries dataframe in prophet ready format
    row - metadata row
    p_uuid_df - dataframe containing all parameter-dict and id
    """
    DAILY_FOURIER_ORDER = 0
    WEEKLY_FOURIER_ORDER = 0

    df_p_dict = {} # store and return performance metrics of models
    
    # submit tasks to the client
    for p_uuid, params_row in p_uuid_df.iterrows(): # we use all_params2 here, which have the weekend param too
        #try:
        params = params_row.to_dict()
        df_p = hparam_tuning(df, params, row,
                                is_weekend = True,
                                parallel = "dask",
                                daily_fourier_order=DAILY_FOURIER_ORDER,
                                weekly_fourier_order=WEEKLY_FOURIER_ORDER)
        #_params = { **params, **{"weekend": True}}
        df_p_dict[ p_uuid ] = df_p

        
    return df_p_dict
    

def resubmit_training(df, row, params_row):
    DAILY_FOURIER_ORDER = 0
    WEEKLY_FOURIER_ORDER = 0
    """
    Submit one model for training in dask
    """

    params = params_row.to_dict()

    df_p = hparam_tuning(df, params, row,
                            is_weekend = True,
                            parallel = "dask",
                            daily_fourier_order=DAILY_FOURIER_ORDER,
                            weekly_fourier_order=WEEKLY_FOURIER_ORDER)

    return df_p


def myForecast(df, tuning_results,
                    start, end, row, score,
                    daily_fourier_order = 0,
                    weekly_fourier_order = 0,
                    is_weekend = False, 
                    country_name = "USA",
                    alpha = 1.0,                    
                    percent=0.1):

    """
    df - dataframe with ts and kpi value (y)
    tuning_results - one rowed df with model metrics as cols, 
                    scores belonging to one parameter combination as vals
    row - row of metadata
    end - pd.Timestamp end of an interval of something #? end of interval for known data
    score - str, name of score
    """
    kpi = row["kpi_name"]
    dim_dict = row["dimension_name"]

    # Choosing the best model. If there are multiple equally good, pick one randomly
    tuning_results[score+'_rank'] = tuning_results[score].rank()
    tuning_results['rank'] = tuning_results[score+'_rank']
    params = tuning_results.loc[ tuning_results["rank"].idxmin(), ["changepoint_prior_scale", "changepoint_range"] ].to_dict()
    # Fit model with best params, predict future
    m = run_prophet_funct(df, params, daily_fourier_order, weekly_fourier_order, is_weekend, country_name)
    future = make_future(m, end, 168)
    forecast = m.predict(future)

    # Setting bounds 
    df["doy"] = df.ds.dt.dayofyear
    iqr = (df[['doy', 'y']].groupby('doy').quantile(0.75)-df[['doy', 'y']].groupby('doy').quantile(0.25)).median().values[0]
    minimum = -iqr
    maximum = +iqr

    lower = kpi_constraints_dict[kpi][0]
    upper = kpi_constraints_dict[kpi][1]


    #? What happens here?
    additive_condition = add_cond_trend_version(forecast, percent, lower, upper, minimum, maximum, end)
    if additive_condition:

        if len(m.changepoints[np.abs(np.nanmean(m.params['delta'], axis=0)) >= 0.01].values)==0:
            last_changepoint = start

        else:
            last_changepoint = m.changepoints[np.abs(np.nanmean(m.params['delta'], axis=0)) >= 0.01].values[-1]

        last_point = ((forecast.set_index('ds')[last_changepoint:]['trend']+alpha*minimum>lower)
                    & (forecast.set_index('ds')[last_changepoint:]['trend']+alpha*maximum<upper))[::-1].idxmax() 

        forecast.loc[forecast['ds']>last_point, 'trend'] = forecast.loc[forecast['ds']==last_point, 'trend'].values[0]

        forecast['yhat'] = forecast['trend']

    # Throw away out of bound predictions
    forecast['yhat'] = forecast['yhat'].clip(lower = lower, upper = upper)

    scaler =  StandardScaler(with_mean = False) # RobustScaler
    scaler.fit(df['y'].values.reshape(-1,1)) 

    df = df.set_index("ds")
    forecast = forecast.set_index("ds")

    results = pd.DataFrame( index = forecast.index,
                            columns = ["kpi_name", "dimension_name", "ground_truth",
                                        "pred", "error", "trend", "gt_wo_trend", "pred_wo_trend"])

    results["kpi_name"] = [kpi] * len(results)
    results["dimension_name"] = [dim_dict] * len(results)
    results["ground_truth"] = df.y
    results["pred"] = forecast.yhat
    results["error"] = scaler.transform((df['y']-forecast['yhat']).values.reshape(-1,1)).T[0]
    results["trend"] = forecast.trend
    results["gt_wo_trend"] = df.y - forecast.trend              # ground truth without trend
    results["pred_wo_trend"] = forecast.yhat - forecast.trend   # predictions without trend

    future_results = results.loc[ ~results.index.isin(df.index)]
    past_results = results.loc[ df.index ]

    return past_results #,future_results




def refresh_zk_table( zookeeper_table, compute, metadata_store,p_uuid_df ):
    status_df = pd.DataFrame(compute).applymap(lambda x: x.status)
    unfinished_tasks = [(status_df.index[x], status_df.columns[y]) for x, y in zip(*np.where(status_df.values != 'finished'))] 

    for _, row in zookeeper_table.iterrows():
        ts_uuid = row.ts_uuid
        p_uuid  = row.p_uuid
        
        if (p_uuid, ts_uuid) not in unfinished_tasks:
            mask = ((ts_uuid == zookeeper_table.ts_uuid) & (p_uuid == zookeeper_table.p_uuid))

            zk_row = row.copy()
            zk_row["failed"] = False
            zookeeper_table[ mask ] = zk_row

    for p_uuid, ts_uuid in unfinished_tasks:

        if (ts_uuid in zookeeper_table.ts_uuid.values) and (p_uuid in zookeeper_table.p_uuid.values):

            mask = ((ts_uuid == zookeeper_table.ts_uuid) & (p_uuid == zookeeper_table.p_uuid))
            zk_row = zookeeper_table[ mask ].copy()
            zk_row.num_failed += 1
            zk_row.last_failed_ts = int(time.time())

            zk_row.bad_status = compute[ts_uuid][p_uuid].status
            zookeeper_table[ mask ] = zk_row

        else:
            mdrow = metadata_store[ metadata_store.ts_uuid == ts_uuid ].copy()

            task_to_retry = { "ts_uuid": ts_uuid, "p_uuid": p_uuid, "ts": mdrow.ts.values[0], 
                                "kpi": mdrow.kpi_name.values[0], "dim": mdrow.dimension_name.values[0],
                                "params": p_uuid_df.loc[p_uuid].to_dict(), "failed": True,
                                "num_failed": 1, "last_failed_ts": int(time.time()),
                                "bad_status": compute[ts_uuid][p_uuid].status }
            zookeeper_table = zookeeper_table.append(pd.Series(task_to_retry), ignore_index=True)
            
    return zookeeper_table

# %%
def retry_unfinished_tasks( compute,p_uuid_df,metadata_store,df_dict ): 
            
    status_df = pd.DataFrame(compute).applymap(lambda x: x.status)
    unfinished_tasks = [(status_df.index[x], status_df.columns[y]) for x, y in zip(*np.where(status_df.values != 'finished'))]  # tuple of (p_uuid, ts_uuid)

    re_taskgroup = {}

    for p_uuid, ts_uuid in unfinished_tasks:
        df = df_dict[ ts_uuid ]
        row = metadata_store[ metadata_store.ts_uuid == ts_uuid ]
        params_row = p_uuid_df.loc[ p_uuid ]

        if ts_uuid not in re_taskgroup:
            re_taskgroup[ts_uuid] = {}
        re_taskgroup[ts_uuid][p_uuid] = dask.delayed(resubmit_training)( df, row, params_row )

    re_compute = dask.compute(re_taskgroup)[0] # recompute previously unfinished tasks

    # Putting back new tasks - they must be mostly finished, there might be still unfinished though, then we need to 
    for ts_uuid, tg in re_compute.items():
        
        for p_uuid, task in tg.items():
            compute[ts_uuid][p_uuid] = re_compute[ts_uuid][p_uuid]
        
    return compute


if __name__ == "__main__":
    main()

