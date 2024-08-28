import dask
from dask import dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score
from dask.distributed import Client
import argparse
import time

print('inside script logistic regression')
start = time.time()
print('start time', start)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path")
parser.add_argument("--max_iter", type=int)
parser.add_argument("--C", type=float)
parser.add_argument("--penalty")
parser.add_argument("--blocksize", default='134217728')
args = parser.parse_args()
input_path = args.input_path
max_iter = args.max_iter
C = args.C
penalty = args.penalty
block_size = args.blocksize
print('input data path ', input_path)
print('max_iter', max_iter)
print('C', C)
print('penalty', penalty)
print('block_size', block_size)
client = Client()

print('chunk size', dask.config.get('array.chunk-size'))
df = dd.read_csv(input_path + "*", blocksize=block_size,
                 storage_options={"key": 'root', "secret": 'password123',
                                  "client_kwargs": {
                                      "endpoint_url":
                                          "http://10.98.146.20:9000"}})

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1),
    df[['label']],
    random_state=0,
    shuffle=False,
    test_size=0.3,
    train_size=0.7)

X_training = X_train.to_dask_array(lengths=True)
y_training = y_train.to_dask_array(lengths=True)


print('training chunk size', X_training.chunksize)
print('training label chunk size', y_training.chunksize)

print('training number of chunks', X_training.chunks)
print('training label number of chunks', y_training.chunks)

lr = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty, solver='lbfgs')

lr = lr.fit(X=X_training, y=y_training)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test.to_dask_array(lengths=True))
y_prediction = y_pred.astype(int)
y_actual = y_test['label'].to_dask_array(lengths=True)

accuracy = accuracy_score(y_actual, y_prediction)
print(accuracy)
end_time = time.time()
print('end time', end_time)
elapsed_time = end_time - start
print(f'total Time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
