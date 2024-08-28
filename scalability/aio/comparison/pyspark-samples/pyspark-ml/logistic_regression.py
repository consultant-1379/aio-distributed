from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import sys

if __name__ == '__main__':
    input_path = sys.argv[1]
    max_iter = int(sys.argv[2])
    reg_param = float(sys.argv[3])
    elastic_net_param = float(sys.argv[4])

    print('max_iter', max_iter)
    print('reg_param', reg_param)
    print('elastic_net_param', elastic_net_param)

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

    train, test = df.randomSplit([0.7, 0.3])

    assembler = VectorAssembler(
        inputCols=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature '
                                                                       '5',
                   'feature 6', 'feature 6', 'feature 7', 'feature 8', 'feature '
                                                                       '9',
                   'feature 10', 'feature 11', 'feature 12', 'feature 13',
                   'feature '
                   '14',
                   'feature 15'
                   ],
        outputCol='features')

    trainingData = assembler.transform(train).select('features', 'label')
    lr = LogisticRegression(maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param)
    lr.setFeaturesCol('features')
    lr.setFamily("binomial")
    # Fit the model
    lrModel = lr.fit(trainingData)

    print("Coefficients: ", lrModel.coefficientMatrix)
    print("Intercept: ", lrModel.interceptVector)

    testingData = assembler.transform(test)
    predictions = lrModel.transform(testingData)
    evaluator = BinaryClassificationEvaluator()
    evaluator.setRawPredictionCol('rawPrediction')
    evaluator.setLabelCol('label')
    metrics = evaluator.evaluate(predictions)
    print(metrics)

    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
    print("Accuracy : ", accuracy)
    spark.stop()
