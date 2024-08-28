from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
import time


start = time.time()
iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# creating dataframe of IRIS dataset
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
                     'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
                     'species': iris.target})

clf = RandomForestClassifier(n_estimators=90000,n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

end = time.time()
print('elapsed time', end - start)

# start time 1666770998.193671
# ACCURACY OF THE MODEL:  0.9777777777777777
# elapsed time without dask  139.5779459476471


