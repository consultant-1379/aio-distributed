from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import sys
import time

if __name__ == '__main__':
    input_path = sys.argv[1]
    start = time.time()
    print('start time', start)
    spark = SparkSession \
        .builder \
        .config("spark.hadoop.fs.s3a.endpoint", "http://10.98.146.20:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "root") \
        .config("spark.hadoop.fs.s3a.secret.key", "password123") \
        .config("spark.hadoop.fs.s3a.path.style.access", True) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    load_options = {'header': True, 'delimiter': ',', 'inferSchema': True}
    df = spark.read.format("csv").options(**load_options).load(
        input_path)

    assembler = VectorAssembler(
        inputCols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
                   ],
        outputCol='features')

    trainingData = assembler.transform(df).select('features')

    kmeans = KMeans().setK(3).setSeed(1).setMaxIter(20).setInitSteps(2).setInitMode('k-means||').setDistanceMeasure(
        'euclidean').setTol(0.0001)

    # Fit the model
    kmeans_model = kmeans.fit(trainingData)

    print("centroids: ", kmeans_model.clusterCenters())
    spark.stop()
    end_time = time.time()
    print('end time', end_time)
    elapsed_time = end_time - start
    print(f'total Time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    # centroids: [array([6.80056572, 3.23938825, 1.31347263, 1.34757302]),
    #             array([6.94094757, 3.33006021, 5.53268585, 1.29141474]),
    #             array([6.62500788, 3.25499979, 3.63821813, 1.36893208])]