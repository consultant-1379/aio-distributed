import pandas as pd
import pmdarima
import scipy
from scipy.stats import friedmanchisquare, kruskal, ks_2samp
from statsmodels.tsa.stattools import acf
#from configs import bad_direction_kpi_dict as dir_dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.stattools import acf
from scipy.spatial import distance_matrix


from .kneebow.rotor import Rotor



# Data related stuff #
def pivot_maker(time_series): # used in tests
    df = pd.DataFrame(time_series)
    df['hour'] = df.index.hour
    df['date'] = df.index.dayofyear

    pivot_table = df.pivot(index='date', columns='hour', values=time_series.name)
    return pivot_table


trim = lambda ser: ser.loc[ser.first_valid_index():ser.last_valid_index()]
#ACF condition
def ACF_condition(series):
    ACF = acf(series, nlags = 24)
    max_dist = distance_matrix(ACF[10:].reshape(-1,1),ACF[10:].reshape(-1,1)).max()
    return max_dist 
#drop parts with a lot of missing data funct

def drop_missing_data(dates, window):
    starting = dates[0]
    ending = dates[-1]
    gaps = np.diff(dates) / np.timedelta64(1, 'h')
    gaps = pd.Series( gaps, index = dates[:-1] )

    mv_avg = gaps.rolling( window, win_type='triang').mean()[ window-1:] # the first few are nan, bc of window
    rotor = Rotor()
    to_rotate = np.empty( shape = (mv_avg.shape[0], 2) )

    to_rotate[:, 0] = mv_avg.index.map(lambda idx: idx.timestamp()) # np.arange( mv_avg.shape[0] )
    to_rotate[:, 1] = mv_avg.sort_values(ascending = False).to_numpy()

    rotor.fit_rotate( to_rotate, scale = False )
    argmin = np.argmin(rotor._data[:,1]) + 3

    mindate = mv_avg.index[ argmin ]


    #missing_series = pd.Series(np.diff(dates)/np.timedelta64(1, 'h'), index=dates[:-1])
    #print('top 5 missing:', missing_series.sort_values(ascending = False).head(10))
    #plt.figure(figsize=(20,5))
    #plt.plot(missing_series.resample('H').asfreq().index, missing_series.resample('H').asfreq().values)
    #plt.show()
    #print('HISTOGRAM')
    #print(missing_series.max())
    #plt.figure(figsize=(20,5))
    #plt.hist(missing_series.resample('H').asfreq().values,bins =int(missing_series.max()) )
    #plt.show()
    if (gaps<=2).all():
        #print('all')
        change= False
        return change, starting, ending
    # wind = window
    # mvng_avg = missing_series.rolling(wind, win_type='triang').mean()

    # moving_averages = mvng_avg[wind-1:] #* eredetileg int(np.sqrt(len(mvng_avg))) a vége

    
    #L_method
    # regr = linear_model.LinearRegression()
    # cluster_num=np.arange(len(moving_averages))   
    # mse=np.zeros(len(moving_averages))
    # xo =np.zeros(len(moving_averages))
    # tofit=moving_averages
    # for i in cluster_num[2:-2]: # TODO legyen sqrt(tofit.shape)
        # #print(i)
        # guess11=regr.fit(cluster_num[i:].reshape(-1,1), tofit[i:].values.reshape(-1, 1)).predict(cluster_num[i:].reshape(-1,1))
        # guess1= regr.fit(cluster_num[i:].reshape(-1,1), tofit[i:].values.reshape(-1, 1))
        # a= guess1.coef_.item()
        # b= guess1.intercept_.item()
        # guess22=regr.fit(cluster_num[:i].reshape(-1, 1),tofit[:i].values.reshape(-1, 1)).predict(cluster_num[:i].reshape(-1, 1))
        # guess2 =regr.fit(cluster_num[:i].reshape(-1, 1),tofit[:i].values.reshape(-1, 1))
        # c= guess2.coef_.item()
        # d= guess2.intercept_.item()
        # mse[i]= (len(moving_averages)-i+1)*mean_squared_error(tofit[i:], guess11 )+i*mean_squared_error(tofit[:i], guess22)
        # #xo[i]=(d-b)/(a-c)
    # minhely=np.argmin(mse[2:-2])+2+1
    
    #plt.plot(moving_averages)
    #plt.scatter([moving_averages.index[minhely]], [1],color='red')
    #plt.show()
    # mindate = moving_averages.index[minhely]
    # print(mindate)          #! debug
    if ending-mindate>pd.Timedelta("2w"):
        starting = mindate
    elif mindate-starting>pd.Timedelta("2w"):
        ending = mindate
    if (starting!=dates[0]) or (ending!=dates[-1]):
        change=True
    else:
        change=False
    return change, pd.to_datetime(starting), pd.to_datetime(ending)
#*        



# Testing related stuff #
def decision(results):
    return sum(results)*2 >= len(results)

# Seasonality
def welch_test(pivot_table):
    tau = pivot_table.shape[1]
    N = pivot_table.count()
    z_hat = pivot_table.mean()

    s_squared = ((pivot_table-z_hat)**2).sum() / (N-1)
    w = N / s_squared
    W = w.sum()
    const = (w * z_hat / W).sum()
    enum = (w*(z_hat-const)**2).sum() / (tau-1)
    denom = 1 + 2 * (tau+2) / (tau**2-1) * ((1-w/W)**2 / (N-1)).sum()
    f = enum / denom
    
    f1 = tau - 1
    f2 = 1 / (3 / (tau**2-1) * ((1-w/W)**2 / (N-1)).sum())
    p = scipy.special.fdtrc(f1, f2, f)
    return f, p


def seas_testing(data_series, alpha, beta):
    data_pivot = pivot_maker(data_series)

    Welch = welch_test(data_pivot)[1] < alpha
    Friedman = friedmanchisquare(*data_pivot.T.values)[1] < alpha
    Kruskal = kruskal(*data_pivot.T.values, nan_policy='omit')[1] < alpha
    #print('WELCH:', welch_test(data_pivot)[1], Welch, 'FRIEDMAN:', friedmanchisquare(*data_pivot.T.values)[1], Friedman,'KRUSKAL:', kruskal(*data_pivot.T.values, nan_policy='omit')[1], Kruskal )
    return Welch, Friedman, Kruskal


def test_seasonality(data_series, alpha=.001, beta=.5, alpha_r=.01, beta_r=.5):
    tests = seas_testing(data_series, alpha, beta)

    tests_res = ()
    if (sum(data_series.isna())==0):
        arima = pmdarima.arima.auto_arima(data_series, seasonal=False, max_p=3, max_q=3)
        residual = pd.Series(arima.resid(), index=data_series.index)
        tests_res += seas_testing(residual, alpha_r, beta_r)
        
    return decision(tests + tests_res)


def pivot_maker_vectorized(time_series): # used in tests
    df = pd.DataFrame(time_series)
    df['hour'] = df.index
    df['hour'] = pd.to_datetime(df['hour'])
    df['date'] = df.hour.dt.dayofyear
    df['hour'] = df["hour"].dt.hour

    pivot_table = df.pivot(index='date', columns='hour', values=pd.Series(time_series).name)
    return pivot_table


def seas_testing_vectorized(data_series, alpha, beta):
    data_pivot = pivot_maker_vectorized(data_series)
    data_pivot = data_pivot.apply(pd.to_numeric)
    
    Welch = welch_test(data_pivot)[1] < alpha
    Friedman = friedmanchisquare(*data_pivot.T.values)[1] < alpha
    try:
        Kruskal = kruskal(*data_pivot.T.values, nan_policy='omit')[1] < alpha
    except ValueError:
        Kruskal = True
    #print('WELCH:', welch_test(data_pivot)[1], Welch, 'FRIEDMAN:', friedmanchisquare(*data_pivot.T.values)[1], Friedman,'KRUSKAL:', kruskal(*data_pivot.T.values, nan_policy='omit')[1], Kruskal )
    return Welch, Friedman, Kruskal


def test_seasonality_vectorized(data_series, alpha=.001, beta=.5, alpha_r=.01, beta_r=.5):
    tests = seas_testing_vectorized(data_series, alpha, beta)

    tests_res = ()
    if (sum(data_series.isna())==0):
        arima = pmdarima.arima.auto_arima(data_series, seasonal=False, max_p=3, max_q=3)
        residual = pd.Series(arima.resid(), index=data_series.index)
        tests_res += seas_testing_vectorized(residual, alpha_r, beta_r)
        
    return decision(tests + tests_res)

# Stationarity
def stat_testing(ser, k, alpha=.02):
    ser = trim(ser) # is this necessary? - look at ser[ser<=quant] below
    n = ser.shape[0]
    last_slice = ser.iloc[(k-1)*n//k:]
    slices = [ser.iloc[i*n//k:(i+1)*n//k].values.T for i in range(k)]
    nem_egyezik = False
    tabla = np.zeros((4,4))
    tabla2 = np.zeros((4,4))
    p_values = np.zeros((4,4))

    for i in range(len(slices)-1):
        for j in range(i+1, len(slices)):
            #print(slices[i][0], slices[j][0])
            #print(slices[i])
            #print(slices[j])
            test = ks_2samp(slices[i],slices[j])
            p_values[i,j] = test.pvalue
            #test2 = scipy.stats.median_test(slices[i][0],slices[j][0]) # p big -> medians are the same
            #print('median p value:',i, j,  test2[1])
            if (test.pvalue < alpha): 
                nem_egyezik = True
                tabla[i,j] =tabla[j,i]=  1
                #print('nem egyezik', i, j)
            #if test2[1]<alpha:
            #    tabla2[i,j]=tabla2[j,i] = 1

    stat = not (list(tabla.sum(axis=1)) not in [[1.,3.,1.,1.], [3,1,1,1], [1,1,3,1]]) & (tabla.sum().sum()>4)
    return stat

    

def test_stationarity(ser, direction, tested_ratio=.4, k=4, window_size=48):
    is_stat = False
    
    if (direction in ['min','both']):
        quant_l = ser.rolling(window_size, center=True).quantile(tested_ratio)
        ser_test = ser[ser<=quant_l]
        n = ser_test.shape[0]
        is_stat |= stat_testing(ser_test, k)
        
    if (direction in ['max','both']):
        quant_u = ser.rolling(window_size, center=True).quantile(1-tested_ratio)
        ser_test = ser[ser>=quant_u]
        
        n = ser_test.shape[0]

        is_stat |= stat_testing(ser_test, k)        

    return is_stat