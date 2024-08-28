# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows,
# actions, and settings.

from sklearn import cluster, datasets
import numpy as np
import pandas as pd
import boto3
from io import StringIO


def push_file(file, bucket_name, file_path):
    try:
        client = boto3.client(
            's3',
            endpoint_url="http://10.196.120.232:31450/",
            aws_access_key_id="root",
            aws_secret_access_key="password123",
        )

        client.put_object(
            Bucket=bucket_name,
            Key=file_path,
            Body=file
        )
    except Exception as e:
        print("Error", e)

def upload_file(file_name, bucket_name, file_path):
    try:
        client = boto3.client(
            's3',
            endpoint_url="http://10.196.120.232:31450/",
            aws_access_key_id="root",
            aws_secret_access_key="password123",
        )

        client.upload_file(
            file_name,
            Bucket=bucket_name,
            Key=file_path
        )
    except Exception as e:
        print("Error", e)


# 4 million
# n_samples = 4000000
# n_samples = 3500000
n_samples = 100
n_features = 15

cluster_centers = [
    [1.39064343, 9.05630583, -5.74491757, 5.68217205, -7.73614625,
     1.30444867,
     -0.30673086, 7.80905995, -3.99503116, 6.41429852, -4.26076689,
     4.16276007,
     -6.36735649, 0.73585534, -6.950509],
    [1.01150024, -0.63431061, -5.38975473, -3.7208553, 9.05503245,
     -4.23280932,
     5.10049926, 5.40404863, -0.15590343, -1.19708902, -7.95594746,
     1.16248411,
     7.42327592, 6.43418329, -4.32828007],
    [2.20066117, 3.08106684, -5.87270565, 8.16243091, 4.88441146,
     -4.36091632,
     -2.23464193, -7.8854348, 8.16490409, 8.55695873, -6.24739162,
     1.00753028,
     2.25060393, -2.08396147, -3.96708705],
    [7.17842275, 4.37454528, -0.65384241, 3.4047682, 2.79863239, -9.7507896,
     0.64502111, 3.15872139, 4.22142134, -2.89507349, 3.74509578,
     8.1584141,
     8.89123432, 8.51640809, -4.47024671],
    [9.40926206, 4.8878462, -6.18150237, -6.03562573, -7.5588567,
     -8.10925762,
     4.97197146, 4.58706367, 4.99220373, 9.8442251, 3.29943891,
     0.6915018,
     -8.08796157, 2.17553172, -2.76184941]
]

cluster_std = 2.0
# random_state = 170
random_state = np.random.RandomState()

for i in range(1, 6):
    X, y, centers = datasets.make_blobs(n_samples=n_samples,
                                        random_state=random_state,
                                        return_centers=True,
                                        centers=cluster_centers,
                                        cluster_std=cluster_std,
                                        n_features=n_features
                                        )

    # print(centers, '\n')
    # print(cluster_centers==centers)
    # print('mean -->', np.mean(X, axis=0), '\n')
    # print('std -->', np.std(X, axis=0), '\n')

    # print('std -->', np.std(X), '\n')

    # print("##########################################", '\n')

    features = pd.DataFrame(X,
                            columns=['feature %d' % (i + 1) for i in range(0,
                                                                           15)])
    features['label'] = y
    features.to_csv(
        'sample_' + str(i) + '.csv', index=False)
    print('file created --> ','sample_' + str(i) + '.csv')

    # csv_buf = StringIO()
    # features.to_csv(csv_buf, index=False)
    # push_file(csv_buf.getvalue(), 'clustering',
    #           '5GB/clustering_input_' + str(i) + '.csv')
    # csv_buf = None


for i in range(1, 6):
    upload_file('sample_' + str(i) + '.csv', 'clustering',
                '5GB/clustering_input_' + str(i) + '.csv')
    print('file uploaded --> ', 'sample_' + str(i) + '.csv')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# https://scikit-learn.org/stable/auto_examples/cluster
# /plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster
# -comparison-py


# pip install -U scikit-learn
