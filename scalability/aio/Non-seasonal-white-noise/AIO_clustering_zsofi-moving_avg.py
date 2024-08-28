#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from collections import OrderedDict
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.preprocessing import RobustScaler

import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()

from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline

from scipy.signal import savgol_filter
from scipy import stats
from scipy import interpolate

from sklearn.neighbors import DistanceMetric


# In[5]:


import math
from shapely.geometry import asPoint
from shapely.geometry import asLineString
from shapely.geometry import asPolygon
from shapely.ops import transform
from functools import partial
import pyproj


# In[1]:


import sys
sys.path.append('..')
from config.bad_direction_kpi_dict import bad_direction_kpi_dict
from config.kneebow.rotor import Rotor
# sama as:
#sys.path.append('../..')
#from BigT.configs import bad_direction_kpi_dict


# In[6]:


# concave hullt kirakni packagebe
def write_line_string(hull):
    with open("data/line_{0}.csv".format(hull.shape[0]), "w") as file:
        file.write('\"line\"\n')
        text = asLineString(hull).wkt
        file.write('\"' + text + '\"\n')


class ConcaveHull(object):

    def __init__(self, points, prime_ix=0):
        if isinstance(points, np.core.ndarray):
            self.data_set = points
        elif isinstance(points, list):
            self.data_set = np.array(points)
        else:
            raise ValueError('Please provide an [N,2] numpy array or a list of lists.')

        # Clean up duplicates
        self.data_set = np.unique(self.data_set, axis=0)

        # Create the initial index
        self.indices = np.ones(self.data_set.shape[0], dtype=bool)

        self.prime_k = np.array([3, 5, 7, 11, 13, 17, 21, 23, 29, 31, 37, 41, 43,
                                 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
        self.prime_ix = prime_ix

    @staticmethod
    def buffer_in_meters(hull, meters):
        proj_meters = pyproj.Proj(init='epsg:3857')
        proj_latlng = pyproj.Proj(init='epsg:4326')

        project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
        project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

        hull_meters = transform(project_to_meters, hull)

        buffer_meters = hull_meters.buffer(meters)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    def get_next_k(self):
        if self.prime_ix < len(self.prime_k):
            return self.prime_k[self.prime_ix]
        else:
            return -1

    def haversine_distance(self, loc_ini, loc_end):
        lon1, lat1, lon2, lat2 = map(np.radians,
                                     [loc_ini[0], loc_ini[1],
                                      loc_end[:, 0], loc_end[:, 1]])

        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1

        a = np.square(np.sin(delta_lat / 2.0)) +             np.cos(lat1) * np.cos(lat2) * np.square(np.sin(delta_lon / 2.0))

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        meters = 6371000.0 * c
        return meters

    @staticmethod
    def get_lowest_latitude_index(points):
        indices = np.argsort(points[:, 1])
        return indices[0]

    def get_k_nearest(self, ix, k):
        """
        Calculates the k nearest point indices to the point indexed by ix
        :param ix: Index of the starting point
        :param k: Number of neighbors to consider
        :return: Array of indices into the data set array
        """
        ixs = self.indices

        base_indices = np.arange(len(ixs))[ixs]
        distances = self.haversine_distance(self.data_set[ix, :], self.data_set[ixs, :])
        sorted_indices = np.argsort(distances)

        kk = min(k, len(sorted_indices))
        k_nearest = sorted_indices[range(kk)]
        return base_indices[k_nearest]

    def calculate_headings(self, ix, ixs, ref_heading=0.0):
        """
        Calculates the headings from a source point to a set of target points.
        :param ix: Index to the source point in the data set
        :param ixs: Indexes to the target points in the data set
        :param ref_heading: Reference heading measured in degrees counterclockwise from North
        :return: Array of headings in degrees with the same size as ixs
        """
        if ref_heading < 0 or ref_heading >= 360.0:
            raise ValueError('The reference heading must be in the range [0, 360)')

        r_ix = np.radians(self.data_set[ix, :])
        r_ixs = np.radians(self.data_set[ixs, :])

        delta_lons = r_ixs[:, 0] - r_ix[0]
        y = np.multiply(np.sin(delta_lons), np.cos(r_ixs[:, 1]))
        x = math.cos(r_ix[1]) * np.sin(r_ixs[:, 1]) -             math.sin(r_ix[1]) * np.multiply(np.cos(r_ixs[:, 1]), np.cos(delta_lons))
        bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0 - ref_heading
        bearings[bearings < 0.0] += 360.0
        return bearings

    def recurse_calculate(self):
        """
        Calculates the concave hull using the next value for k while reusing the distances dictionary
        :return: Concave hull
        """
        recurse = ConcaveHull(self.data_set, self.prime_ix + 1)
        next_k = recurse.get_next_k()
        if next_k == -1:
            return None
        # print("k={0}".format(next_k))
        return recurse.calculate(next_k)

    def calculate(self, k=3):
        """
        Calculates the convex hull of the data set as an array of points
        :param k: Number of nearest neighbors
        :return: Array of points (N, 2) with the concave hull of the data set
        """
        if self.data_set.shape[0] < 3:
            return None

        if self.data_set.shape[0] == 3:
            return self.data_set

        # Make sure that k neighbors can be found
        kk = min(k, self.data_set.shape[0])

        first_point = self.get_lowest_latitude_index(self.data_set)
        current_point = first_point

        # Note that hull and test_hull are matrices (N, 2)
        hull = np.reshape(np.array(self.data_set[first_point, :]), (1, 2))
        test_hull = hull

        # Remove the first point
        self.indices[first_point] = False

        prev_angle = 270    # Initial reference id due west. North is zero, measured clockwise.
        step = 2
        stop = 2 + kk

        while ((current_point != first_point) or (step == 2)) and len(self.indices[self.indices]) > 0:
            if step == stop:
                self.indices[first_point] = True

            knn = self.get_k_nearest(current_point, kk)

            # Calculates the headings between first_point and the knn points
            # Returns angles in the same indexing sequence as in knn
            angles = self.calculate_headings(current_point, knn, prev_angle)

            # Calculate the candidate indexes (largest angles first)
            candidates = np.argsort(-angles)

            i = 0
            invalid_hull = True

            while invalid_hull and i < len(candidates):
                candidate = candidates[i]

                # Create a test hull to check if there are any self-intersections
                next_point = np.reshape(self.data_set[knn[candidate]], (1,2))
                test_hull = np.append(hull, next_point, axis=0)

                line = asLineString(test_hull)
                invalid_hull = not line.is_simple
                i += 1

            if invalid_hull:
                return self.recurse_calculate()

            # prev_angle = self.calculate_headings(current_point, np.array([knn[candidate]]))
            prev_angle = self.calculate_headings(knn[candidate], np.array([current_point]))
            current_point = knn[candidate]
            hull = test_hull

            # write_line_string(hull)

            self.indices[current_point] = False
            step += 1

        poly = asPolygon(hull)

        count = 0
        total = self.data_set.shape[0]
        for ix in range(total):
            pt = asPoint(self.data_set[ix, :])
            if poly.intersects(pt) or pt.within(poly):
                count += 1
            else:
                d = poly.distance(pt)
                if d < 1e-5:
                    count += 1

        if count == total:
            return hull
        else:
            return self.recurse_calculate()


# In[7]:


def read_filter_metadata(path, seasonality = False):
    
    metadata = pd.read_csv(path ,index_col = 0)
    metadata["seasonality_flag"] = metadata["is_seasonal"].copy()
    metadata['statonarity_flag'] = True
    ### no seasonality or trend:
    metadata = metadata[(metadata['seasonality_flag'] == seasonality) & (metadata['statonarity_flag'] == True)] 
    metadata.index = range(metadata.shape[0])
    
    return metadata


# # Reading the file from the path

# In[8]:


def read_data_and_create_dataframe(metadata,kpi,dimension,path):
    data = pd.read_csv(path)

    data['dt'] = pd.to_datetime(data['dt'])
    
    last_dt = data.dt.max()
    data["dt_max"] = last_dt
    data["dt_max"] = data["dt_max"] - pd.to_timedelta(7, unit='d')
    data["to_keep"] = data.dt>= data.dt_max
    data = data[data.to_keep==True]
    data = data.drop(["to_keep", "dt_max"], axis=1)

    dimension_values = np.unique(metadata[metadata["kpi"]==kpi]["label"])
    
    time_dimension = []
    kpi_values_all_labels = []
    original = []
    dimension_id = []
    
    for dim_value in dimension_values:
        kpi_values = data[data[dimension] == dim_value][kpi].fillna(0)
        original.extend(kpi_values)
        dimension_id.extend([dim_value]*len(kpi_values) )
        scaler = RobustScaler() 

        kpi_values = scaler.fit_transform(kpi_values.values.reshape(-1,1))
        kpi_scale = scaler.scale_
        kpi_center = scaler.center_
        
        time_values = range(kpi_values.shape[0]) 
        ## interquartile rangel megprobalni, beskalazni az idotengelyt, hogy  a tavolsagmetrika jol mukodjon
        time_dimension.extend(time_values)
        kpi_values_all_labels.extend(kpi_values)
    
    X = pd.DataFrame()
    X['time_val'] = time_dimension
    X['kpi_val'] = np.array(kpi_values_all_labels).reshape(-1,1)#kpi_values_all_labels
    X['kpi_original'] = np.array(original).reshape(-1,1)
    X['dim_id'] = np.array(dimension_id).reshape(-1,1)
    

    #X['time_orig'] = X['time_val']
    X['time_val'] = X['time_val'] / 100 #ezen elgondolgozni, ez a skalazas kell-e
    
    return X


# In[9]:


def find_elbow(data, theta):

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
  #  print(rotated_vector.min())

    return np.where(rotated_vector == rotated_vector.min())[0][0]


def find_elbow2(data, theta):

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
   # print(rotated_vector.min())
    return np.where(rotated_vector == rotated_vector.min())[0][0], rotated_vector


def get_data_radiant(data):
    return np.arctan2(data[:, 1].max() - data[:, 1].min(), 
                    data[:, 0].max() - data[:, 0].min())


# In[11]:


def get_opt_eps(data, num_neigh):
    nbrs = NearestNeighbors(n_neighbors=num_neigh, metric='euclidean', n_jobs=-1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = distances[:,num_neigh-1]
    distances = np.sort(distances, axis=0)

    dst_range = np.array([i for i in range(1,len(distances)+1)])
    #zipped = np.column_stack((dst_range,distances[::-1]))
    data_curve = np.dstack([dst_range, distances])[0]
    elbow_index = find_elbow(data_curve, get_data_radiant(data_curve))
    idx_opt = len(data_curve)-elbow_index
    angle = np.rad2deg(get_data_radiant(data_curve))
    index, rotated_data = find_elbow2(data_curve, get_data_radiant(data_curve))
    opt_eps = data_curve[elbow_index, 1]
    
    return opt_eps


# # SVM hyperparameter tunningolva

# In[12]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# In[13]:


def balanced_SVM(data, X_label, kpi):
    X_predictor = data[['time_val', 'kpi_val']]
    over = SMOTE(sampling_strategy=0.7)
    X_predictor, X_label = over.fit_resample(X_predictor, X_label)
    X_train, X_test, y_train, y_test = train_test_split(X_predictor, np.ravel(X_label), 
                                                        test_size = 0.20, random_state = 101)
    #treshold for num timeseries in data
    num_TS = len(data.dim_id.unique())
    if num_TS < 6:
        print(kpi + ' ' + str(num_TS))
        return None
    
    #parameter tunning:
    steps = [('scaler', StandardScaler()), ('SVM', SVC(class_weight = 'balanced'))]
    pipeline = Pipeline(steps) # define the pipeline object
    
    #model = SVC(class_weight = 'balanced')
    param_grid = {'SVM__C': [1], 
              'SVM__gamma': [15,25,30, 40],
              'SVM__kernel': ['rbf']} 
    grid = GridSearchCV(pipeline, param_grid, refit = True, verbose = 3)
    grid.fit(X_train, y_train)
    
    #plot:
    x_min, x_max = X['time_val'].min(), X['time_val'].max() 
    y_min, y_max = X['kpi_val'].min() - 1, X['kpi_val'].max() + 1

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = grid.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=[12,8])
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    # Plot also the training points
    plt.scatter(data['time_val'], data['kpi_val'], c=data['merged'], cmap=plt.cm.coolwarm, s=2)
    plt.xlabel('time')
    plt.ylabel(str(kpi))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    days=['Monday', 'Tuesday','Wednesday', "Thursday",'Friday', 'Saturday','Sunday']
    S=max(data.time_val)
    plt.xticks(ticks=[i for i in np.arange(S/7/2,S, S/7)], labels=days)

    plt.title(str(grid.best_estimator_))

   # plt.savefig(str(kpi)+'balanced')
    plt.show()
    return grid.best_estimator_


# In[15]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where


# In[173]:


def plot_decision_regions(X, y, classifier, title='Classification', kpi='KPI', h=0.02):
    data = pd.DataFrame(X)
    x_min, x_max = data[0].min(), data[0].max() 
    y_min, y_max = data[1].min() - 1, data[1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=[12,8])
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    # Plot also the training points
    plt.scatter(data[0], data[1], c=y, cmap=plt.cm.coolwarm, s=2)
    plt.xlabel('time')
    plt.ylabel(kpi)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(title)

    plt.show()


# # Moving AVG

# In[310]:


def prep_X_y(X, window = 7):
    main_mean = X[X.merged==1].kpi_val.mean()
    out_mean = X[X.merged==0].kpi_val.mean()
    if main_mean > out_mean:
        X_main = X[X.merged==1][['time','time_val', 'kpi_val']].groupby(by='time', 
                                                                    as_index=False).min()
        X_main = X_main[['time_val', 'kpi_val']]
    else:
        X_main = X[X.merged==1][['time','time_val', 'kpi_val']].groupby(by='time', 
                                                                    as_index=False).max()
        X_main = X_main[['time_val', 'kpi_val']]

    X_out = X_main.copy()
    X_main['merged'] = 1
    X_out['merged'] = 0
    
    X_out['kpi_diff'] = X_out['kpi_val']
    X_out['kpi_val'] = X_main['kpi_val'].rolling(window = window).mean()
    shift = int(-(np.floor(window/2)))
    X_out['kpi_val'] = X_out.kpi_val.shift(shift)
    X_out = X_out.dropna()
    X_out['kpi_diff'] = abs(X_out.kpi_val - X_out.kpi_diff)
    try:
        max_diff = max(X_out['kpi_diff'])
    except:
        max_diff = np.std(X_main.kpi_val)

    X_border = X_out.copy()
    X_border['merged'] = 1
    
    if main_mean > out_mean:
        X_out.kpi_val -= max_diff        
    else:
        X_out.kpi_val += max_diff


    X_predictor = pd.concat([X, X_out, X_border])[['time_val', 'kpi_val']]
    X_label = pd.concat([X, X_out, X_border])[['merged']]
    return X_predictor, X_label


# In[176]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
knn.fit(X_predictor, X_label)


# In[177]:


plot_decision_regions(X_predictor.values, X_label.values, knn, title='5 nearest neighbors classification')


# In[301]:


for kpi in balint_adat.kpi_name.unique()[0:5]:
    data = balint_adat[balint_adat.kpi_name == kpi]
    X = data_to_dbscan(data)
    X['merged'] = merge_clusters(X)
    X_predictor, X_label = prep_X_y(X)
    oversample = SMOTE(sampling_strategy=0.6)
    X_pred, X_lab = oversample.fit_resample(X_predictor, X_label)
    
    model1 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan'))        
    model1.fit(X_pred, X_lab)    
    plot_decision_regions(X_pred.values, X_lab.values, model1, title='5 nearest neighbors classification', kpi=kpi)
    
    model2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean'))        
    model2.fit(X_pred, X_lab)    
    plot_decision_regions(X_pred.values, X_lab.values, model2, title='5 nearest neighbors classification', kpi=kpi)


# In[ ]:


for kpi in balint_adat.kpi_name.unique()[5:]:
    data = balint_adat[balint_adat.kpi_name == kpi]
    X = data_to_dbscan(data)
    X['merged'] = merge_clusters(X)
    X_predictor, X_label = prep_X_y(X)
    oversample = SMOTE(sampling_strategy=0.6)
    X_pred, X_lab = oversample.fit_resample(X_predictor, X_label)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean'))
        
    model.fit(X_pred, X_lab)
    
    plot_decision_regions(X_pred.values, X_lab.values, model, title='5 nearest neighbors classification')


# In[240]:


for kpi in balint_adat.kpi_name.unique()[0:5]:
    data = balint_adat[balint_adat.kpi_name == kpi]
    X = data_to_dbscan(data)
    X['merged'] = merge_clusters(X)
    X_predictor, X_label = prep_X_y(X)
    oversample = SMOTE(sampling_strategy=0.6)
    X_pred, y_lab = oversample.fit_resample(X_predictor, X_label)
    knn = KNeighborsClassifier(n_neighbors = 6, metric = 'euclidean')
    knn.fit(X_pred, y_lab)
    
    plot_decision_regions(X_pred.values, y_lab.values, knn, title='5 nearest neighbors classification')


# In[266]:


# function to reduce the number of clusters to two: main cluster, outliers
# input: is a pd.DataFrame, with columns: 'time_val', 'kpi_val', 'labels'
# output: new labels 

def merge_clusters(data, kpi=None):
    data = data.set_index(pd.Index([i for i in range(0, len(data))]))
    labels_ = np.asarray(data['labels'])
    data = data[['time_val','kpi_val']]    
    distinct_labels,counts = np.unique(labels_,return_counts=True)
    main_cluster_id = distinct_labels[np.argmax(counts)]
    
    #if no outliers found=> every point belongs to main cluster                                  
    if data[labels_ == -1].shape[0] == 0:              
        labels_[:] = main_cluster_id
        return labels_
    
    tree_out = KDTree(data[labels_ == -1], leaf_size=2)    
    tree_main = KDTree(data[labels_ == main_cluster_id], leaf_size=2)
    
    #check not too big outliers:
    check_outlier_quantile = np.quantile(data.kpi_val,0.975)
    outliers_to_postprocess = data[(labels_ == -1) & (data['kpi_val'] < check_outlier_quantile)]
    if len(outliers_to_postprocess) != 0:
        dist, ind = tree_main.query(outliers_to_postprocess, k=min(data[labels_ == main_cluster_id].shape[0],5))
        dist_main = np.mean(dist,axis=1)
        dist, ind = tree_out.query(outliers_to_postprocess, k=min(data[labels_ == -1].shape[0],5))
        dist_out = np.mean(dist,axis=1)

        new_label = (dist_main < dist_out)
        new_label = [main_cluster_id if boolean is True else -1 for boolean in new_label]
        labels_[outliers_to_postprocess.index] = new_label
       
    #keep outliers only one side of main cluster
    main_cluster_median = np.median( data[labels_ == main_cluster_id].kpi_val)
    outliers_above = data[(labels_ == -1) & (data['kpi_val'] > main_cluster_median)]
    outliers_under = data[(labels_ == -1) & (data['kpi_val'] < main_cluster_median)]
    if not kpi:
        if len(outliers_above)>0 and len(outliers_under)>0:
            if len(outliers_above) > len(outliers_under):
                labels_[outliers_under.index] = main_cluster_id
            else:
                labels_[outliers_above.index] = main_cluster_id
    elif kpi == 'min':
        labels_[outliers_above.index] = main_cluster_id
    elif kpi == 'max':
        labels_[outliers_under.index] = main_cluster_id
            
    #find the order of cluster to merge, if we have
    if len(distinct_labels)>2:
        data["labels_"] = labels_
        rank_df = data[["kpi_val", "labels_"]][(data.labels_!=main_cluster_id)&(data.labels_!=-1)].groupby(["labels_"]).mean()
        rank_df = np.abs(rank_df).sort_values(by= "kpi_val", ascending=True)
        
        data.drop('labels_', axis=1, inplace=True)
        clusters_to_process = list(rank_df.index)
    else:
        return labels_

    cluster_number_outlier_points  = data[labels_ == -1]
    for i in clusters_to_process:  
                
        ith_data = data[labels_ == i]

        ith_data_mean = ith_data.kpi_val.mean()
        ith_data_max = ith_data.time_val.max()
        ith_data_min = ith_data.time_val.min()
        main_cluster_mean = data[labels_ == main_cluster_id].kpi_val.mean()
        
        tree_out = KDTree(data[labels_ == -1], leaf_size=2)   
        tree_main = KDTree(data[labels_ == main_cluster_id], leaf_size=2)

        if main_cluster_mean>ith_data_mean:
            outliers = cluster_number_outlier_points[(cluster_number_outlier_points.kpi_val<main_cluster_mean)&(cluster_number_outlier_points.kpi_val>ith_data_mean)&(cluster_number_outlier_points.time_val<ith_data_max)&(cluster_number_outlier_points.time_val>ith_data_min)]
        elif main_cluster_mean<ith_data_mean:
            outliers = cluster_number_outlier_points[(cluster_number_outlier_points.kpi_val>main_cluster_mean)&(cluster_number_outlier_points.kpi_val<ith_data_mean)&(cluster_number_outlier_points.time_val<ith_data_max)&(cluster_number_outlier_points.time_val>ith_data_min)]
        for idx, row in outliers.iterrows():
            dist, ind = tree_main.query(np.asarray([row['time_val'], row['kpi_val']]).reshape(1,-1), k=min(data[labels_ == main_cluster_id].shape[0],3))
            dist_main = np.mean(dist,axis=1)
            dist, ind = tree_out.query(np.asarray([row['time_val'], row['kpi_val']]).reshape(1,-1), k=min(data[labels_ == -1].shape[0],3))
            dist_out = np.mean(dist,axis=1)
            if dist_out >= dist_main:
                labels_[idx] = main_cluster_id
            else:
                labels_[idx] = -1
                
        try:
          
            concave_hull = ConcaveHull(ith_data.values)
            hull_array = concave_hull.calculate()

            dist, ind = tree_main.query(hull_array, k=min(data[labels_ == main_cluster_id].shape[0],5))
            dist_main = np.mean(dist,axis=1)
            dist, ind = tree_out.query(hull_array, k=min(data[labels_ == -1].shape[0],5))
            dist_out = np.mean(dist,axis=1)

            new_label = np.sum((dist_main < dist_out)) #hanyszor van kozelebb a foklaszterhez
           # if new_label >= (len(hull_array) / 2):
            if new_label >= (5 / 2):
                new_label = main_cluster_id
            else:
                new_label = -1
            labels_[ith_data.index] = new_label
            
        except:
            ith_data_mean = ith_data.kpi_val.mean()
            outlier_mean = data[labels_ == -1].kpi_val.mean()
            if abs(ith_data_mean-outlier_mean) < abs(ith_data_mean-main_cluster_mean):
                labels_[ith_data.index] = -1
            else:
                labels_[ith_data.index] = main_cluster_id
            
    labels_ = (labels_ != -1) ## outliers are 0 inliers are 1
    return labels_.astype(int)


# # Data by Balint

# ### Balint adata mas formatumu mint az eddigi, igy irtam neki egy uj fuggvenyt, ami feldolgozza

# In[18]:


balint_adat = pd.read_csv('/home/jovyan/work/MAXIMUUUUUS/past_result.csv')
balint_adat.head()


# In[268]:


#ez a fugveny a balint adatat kpi-re szurve elokesziti es DBSCN-eli
def data_to_dbscan(data):

    data['dt'] = pd.to_datetime(data['ds'])

    last_dt = data.dt.max()
    data["dt_max"] = last_dt
    data["dt_max"] = data["dt_max"] - pd.to_timedelta(7, unit='d')
    data["to_keep"] = data.dt>= data.dt_max
    data = data[data.to_keep==True]
    data = data.drop(["to_keep", "dt_max"], axis=1)

    dimension_values = np.unique(data.dimension_name)

    time_dimension = []
    kpi_values_all_labels = []
    original = []
    dimension_id = []

    for dim_value in dimension_values:
        kpi_values = data[data.dimension_name==dim_value]['gt_wo_trend'].fillna(0)
        original.extend(kpi_values)
        dimension_id.extend([dim_value]*len(kpi_values))
        scaler = RobustScaler() 

        kpi_values = scaler.fit_transform(kpi_values.values.reshape(-1,1))
        kpi_scale = scaler.scale_
        kpi_center = scaler.center_

        time_values = range(kpi_values.shape[0]) 
        time_dimension.extend(time_values)
        kpi_values_all_labels.extend(kpi_values)

    X = pd.DataFrame()
    X['time'] = time_dimension
    X['time_val'] = time_dimension
    X['kpi_val'] = np.array(kpi_values_all_labels).reshape(-1,1)#kpi_values_all_labels
    X['kpi_original'] = np.array(original).reshape(-1,1)
    X['dim_id'] = np.array(dimension_id).reshape(-1,1)
    
    kpi_max = np.percentile(X.kpi_val, 95)
    kpi_min = np.percentile(X.kpi_val, 5)
    scal = kpi_max - kpi_min
    length = len(X)
    uni = np.random.uniform(0,1,length)
    X.time_val += uni
    X['time_val'] = X['time_val'].apply(lambda x: (x/163)*scal)
    data = X[['time_val','kpi_val']]
    opt_eps = get_opt_eps(data, 5)
    
    cluster = DBSCAN(eps=opt_eps, min_samples=5, metric='euclidean', n_jobs=-1).fit(np.array(data))
    labels = cluster.labels_
    X['labels'] = labels
    
    return X


# In[ ]:




