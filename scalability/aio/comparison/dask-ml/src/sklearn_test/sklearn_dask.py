from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
import time
import joblib

start = time.time()
print('start time', start)
from dask.distributed import Client

client = Client()
iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# creating dataframe of IRIS dataset
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
                     'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
                     'species': iris.target})

clf = RandomForestClassifier(n_estimators=90000, n_jobs=-1)
with joblib.parallel_backend('dask'):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

end = time.time()
print('elapsed time', end - start)

# 41 seconds gain

# start time 1666771444.783779
# ACCURACY OF THE MODEL:  0.9111111111111111
# elapsed time dask  98.44139909744263
