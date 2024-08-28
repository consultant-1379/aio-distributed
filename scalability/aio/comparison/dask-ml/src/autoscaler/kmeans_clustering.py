from dask import dataframe as dd
from dask_ml.cluster import KMeans
import argparse
from dask.distributed import Client

print('inside script kmeans clustering')

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
args = parser.parse_args()
input_path = args.input_path
print('input data path ', input_path)
client = Client('mycluster-scheduler.dask-operator.svc.cluster.local:8786')
df = dd.read_csv(input_path + "*",
                 storage_options={"key": 'root', "secret": 'password123',
                                  "use_listings_cache": False,
                                  "client_kwargs": {
                                      "endpoint_url":
                                          "http://10.98.146.20:9000"}})

X_tain_dask_array = df.drop('target', axis=1).to_dask_array(lengths=True)
kmeans_model = KMeans(n_clusters=3, random_state=1, max_iter=20, init_max_iter=2, init='k-means||', tol=0.0001)

print('training chunk size', X_tain_dask_array.chunksize)
print('training number of chunks', X_tain_dask_array.chunks)

kmeans_model.fit(X=X_tain_dask_array)

print('cluster centers', kmeans_model.cluster_centers_)
print('labels', kmeans_model.labels_)
print('steps to reach convergence', kmeans_model.n_iter_)

# cluster centers [[6.49751690e+06 6.78020863e+00 3.26936909e+00 3.24849806e+00
#   1.33941174e+00]
#  [1.08320915e+07 6.78022008e+00 3.26951639e+00 3.24861849e+00
#   1.33931512e+00]
#  [2.16542490e+06 6.78024850e+00 3.26952297e+00 3.24897014e+00
#   1.33928431e+00]]
# labels dask.array<astype, shape=(130000000,), dtype=int32, chunksize=(776374,), chunktype=numpy.ndarray>
# steps to reach convergence 20
