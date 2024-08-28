
import pandas as pd
import numpy as np

import prophet
import sys

sys.modules['fbprophet'] = prophet

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import generate_cutoffs, single_cutoff_forecast

from sklearn.preprocessing import StandardScaler

from supersmoother import SuperSmoother
import scipy

import logging
logger = logging.getLogger('prophet')

def flatten(lst): 
    return [element in sublist for sublist in lst]

def is_weekday(timestamp):
    date = pd.to_datetime(timestamp)
    return (date.dayofweek < 5)


def run_prophet_funct(df, params, daily_fourier_order, 
                        is_weekend, weekly_fourier_order, country_name):
    
    """if "weekend" in params:
        is_weekend = params["weekend"]
        params.pop("weekend")
    else:
        is_weekend = False"""
    is_weekend = params.pop("weekend", None)

    if is_weekend:
        df['weekday'] = df['ds'].apply(is_weekday)
        df['weekend'] = ~df['ds'].apply(is_weekday)

        m = Prophet(**params, 
                    daily_seasonality = False, 
                    weekly_seasonality = weekly_fourier_order,
                    uncertainty_samples = 0
                    )

        #hozzaadjuk a holidayeket
        #weekday, weekend
        m.add_seasonality(name='weekday', period=1, fourier_order=daily_fourier_order, condition_name='weekday')
        m.add_seasonality(name='weekend', period=1, fourier_order=daily_fourier_order, condition_name='weekend')
    else:
        m = Prophet(**params, 
                    daily_seasonality = daily_fourier_order, 
                    weekly_seasonality = weekly_fourier_order,
                    uncertainty_samples = 0                    
                    )

    m.add_country_holidays(country_name=country_name)
    m = m.fit(df)
    return m
    
    
def mult_cond(forecast, df, percent, lower, upper, end):

    days = list(set(forecast.ds.dt.dayofyear))
    full_day = days[-2]
    full_day_data = forecast.loc[forecast.ds.dt.dayofyear==full_day]['yhat']
    data = df.copy()
    data['day'] = data.ds.dt.dayofyear
    last_std = full_day_data.std()
    number_of_days = 5
    days_std_table = data.groupby('day').std()['y']

    days= [ day for day in days if len(data.loc[data['day']==day]['y']) > 20 ]

    greater_variance=list(days_std_table[days]<last_std)

    levene=[scipy.stats.levene(full_day_data, data.loc[data['day']==day]['y'])[1]>0.05 for day in days]

    x_day_condition =np.sum(np.logical_or(greater_variance,levene))<=number_of_days

    montone_decreasing_condition = forecast.set_index('ds')[end:]['trend'].diff().values[-1]<0

    negative_trend = forecast.set_index('ds')[end:]['trend'].values[-1]<0
    condition_upper_a = (forecast.set_index('ds')[end:]['yhat']>upper).sum()/len(forecast.set_index('ds')[end:]['yhat'])> percent
    condition_lower_a = (forecast.set_index('ds')[end:]['yhat']<lower).sum()/len(forecast.set_index('ds')[end:]['yhat'])> percent
    return ((montone_decreasing_condition) and (x_day_condition)) or (negative_trend) or (condition_upper_a) or (condition_lower_a)
    
    
def add_cond(forecast, percent, lower, upper):

    condition_lower_a = (forecast.set_index('ds')[end:]['yhat']<lower).sum()/len(forecast.set_index('ds')[end:])> percent
    #the trend is montone decreasing
    condition_lower_b = forecast.set_index('ds')[end:]['trend'].diff().values[-1]<0

    condition_upper_a = (forecast.set_index('ds')[end:]['yhat']>upper).sum()/len(forecast.set_index('ds')[end:]['yhat'])> percent
    #the trend is montone increasing
    condition_upper_b = forecast.set_index('ds')[end:]['trend'].diff().values[-1]>0
    return (condition_upper_a & condition_upper_b) or (condition_lower_a & condition_lower_b)


def add_cond_trend_version(forecast, percent, lower, upper, minimum, maximum, end):

    condition_lower_a = (forecast.set_index('ds')[end:]['yhat']+minimum<lower).sum()/len(forecast.set_index('ds')[end:])> percent
    #the trend is montone decreasing
    condition_lower_b = forecast.set_index('ds')[end:]['trend'].diff().values[-1]<0

    condition_upper_a = (forecast.set_index('ds')[end:]['yhat']+maximum>upper).sum()/len(forecast.set_index('ds')[end:]['yhat'])> percent
    #the trend is montone increasing
    condition_upper_b = forecast.set_index('ds')[end:]['trend'].diff().values[-1]>0
    return (condition_upper_a & condition_upper_b) or (condition_lower_a & condition_lower_b)


def make_future(model, end, periods):
    """
    end: given in pd.datetime format: not unix timestamp, we start our dataframe at end+1h
    model: the prophet model
    period: how many hours to forecast
    """

    dates = pd.date_range(start=end+pd.Timedelta('1H'), end = end+pd.Timedelta(str(periods)+'H'), freq = 'H')
    dates = np.concatenate((np.array(model.history_dates), dates))

    future = pd.DataFrame({'ds': dates})
    return future


def hparam_tuning(df, params, row, parallel=None,
                    daily_fourier_order = 0,
                    weekly_fourier_order = 0,
                    is_weekend = False, 
                    country_name = "USA"
                    ):

    
    init_horizon = (df.ds.max() - df.ds.min()).days 
    
    cv_fun = cross_validation  

    if parallel == "dask":
        cv_fun = my_cross_validation   

    m = run_prophet_funct(df, params, 
                            daily_fourier_order,
                            is_weekend,
                            weekly_fourier_order,
                            country_name)
    
    if float(row['missing_data_ratio_all'])<0.15:
        df_cv = cv_fun(m,initial='16 days', 
                        horizon= '2 days', 
                        period = '2 days',
                        parallel=parallel
                        )

    elif init_horizon < 18 and init_horizon > 14:
        df_cv = cv_fun(m, initial="14 days",
                        horizon="1 days",
                        period ="1 days",
                        parallel=parallel
                        )

    elif init_horizon < 14:
        #TODO: very nagy gáz
        df_cv = pd.DataFrame()

    else:
        df_cv = cv_fun(m,initial='16 days', 
                        horizon= '1 days', 
                        period = '1 days',
                        parallel=parallel
                        )    

    if parallel == None:
        df_cv = performance_metrics(df_cv, rolling_window=1)

    return df_cv

def sSmoothing(df):
    df["range"] = df.index
    max_range = df.range.max()
    
    model = SuperSmoother()
    model.fit(np.array(df.range), df.y, (np.ones(max_range+1)))
    
    tfit = np.linspace(0, max_range, max_range+1)
    yfit = model.predict(tfit)
    df["ytop"] = df["y"].copy()
    df.y = df.ytop - yfit
    
    q3, q1 = np.percentile(df.y, [75 ,25])
    IQR = q3 - q1
    df["y"] = np.where(((df.y < q1-3*IQR)|(df.y > q3+3*IQR)), np.nan, df.y)
    df["y"] = df.y.interpolate(method='akima')
    df.y = df.y + yfit
    df = df.drop(['range'], axis=1)
    
    return df


def sSmoothing(df):
    df["range"] = df.index
    max_range = df.range.max()
    
    model = SuperSmoother()
    model.fit(np.array(df.range), df.y, (np.ones(max_range+1)))
    
    tfit = np.linspace(0, max_range, max_range+1)
    yfit = model.predict(tfit)
    df["ytop"] = df["y"].copy()
    df.y = df.ytop - yfit
    
    q3, q1 = np.percentile(df.y, [75 ,25])
    IQR = q3 - q1
    df["y"] = np.where(((df.y < q1-3*IQR)|(df.y > q3+3*IQR)), np.nan, df.y)
    df["y"] = df.y.interpolate(method='akima')
    df.y = df.y + yfit
    df = df.drop(['range'], axis=1)
    
    return df

def preprocess_data(datas, row, 
                    start, end, ssmoothing = True):
                    
    kpi = row["kpi_name"]
    dim_dict = row["dimension_name"]

    data = datas[(datas[list(dim_dict)] == pd.Series(dim_dict)).all(axis=1)]
    data = data.dropna( subset = [kpi] ) # dropna: axis = 0 removed - 
                                         # not supported in dask, default anyway
    df = pd.DataFrame()
    df['y'] = data[kpi]
    df['ds'] = pd.to_datetime(data["ts"], unit='s')

    df = df.loc[(df['ds']>=start)&(df['ds']<=end)]
    df = df.sort_values('ds')
    df = df.reset_index(drop=True)
    if ssmoothing:
        df = sSmoothing(df)
    return df


def _params_to_df_p(model, df):
    """
        Used when adding changepoint_prior_scale and changepoint_range params to 
        performance dataframe in prophet cross-validation.

        model - fbprophet.Prophet
        df    - pd.DataFrame: output of fbprophet.diagnostics.performance_metrics.
    """

    df["changepoint_prior_scale"] = model.changepoint_prior_scale
    df["changepoint_range"] = model.changepoint_range
    
    return df



def my_cross_validation(model, horizon, 
                        period=None, initial=None, 
                        parallel=None, cutoffs=None, 
                        disable_tqdm=False):




    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)

    predict_columns = ['ds', 'yhat']
    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])
        
    # Identify largest seasonality period
    period_max = 0.
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = pd.Timedelta(str(period_max) + ' days')

    if cutoffs is None:
        # Set period
        period = 0.5 * horizon if period is None else pd.Timedelta(period)

        # Set initial
        initial = (
            max(3 * horizon, seasonality_dt) if initial is None
            else pd.Timedelta(initial)
        )

        # Compute Cutoffs
        cutoffs = generate_cutoffs(df, horizon, initial, period)

    else: 
        raise Exception("Unexpected: cutoff should be None") 

    if initial < seasonality_dt:
            msg = 'Seasonality has period of {} days '.format(period_max)
            msg += 'which is larger than initial window. '
            msg += 'Consider increasing initial.'
            logger.warning(msg)

    try:
        from dask.distributed import get_client
    except ImportError as e:
        raise ImportError("parallel='dask' requires the optional "
                            "dependency dask.") from e
    pool = get_client()
    # delay df and model to avoid large objects in task graph.
    df, model = pool.scatter([df, model])
    

    iterables = ((df, model, cutoff, horizon, predict_columns)
                     for cutoff in cutoffs)
    iterables = zip(*iterables)


    logger.info("Applying in parallel with %s", pool)
    
    predicts = pool.map(single_cutoff_forecast, *iterables)  

    myconcat = lambda predicts: pd.concat(predicts, axis=0).reset_index(drop=True)
    my_perf_metr = lambda df_cv: performance_metrics(df_cv, rolling_window=1)   

    fut = pool.submit(myconcat, predicts)
    df_p_fut = pool.submit(my_perf_metr, fut)
    #df_p_fut = pool.submit(_params_to_df_p, model, df_p_fut)

       
    return df_p_fut    