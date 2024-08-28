FROM gcr.io/spark-operator/spark-py:v3.1.1

ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.4/hadoop-aws-2.7.4.jar $SPARK_HOME/jars
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar $SPARK_HOME/jars
ADD https://repo1.maven.org/maven2/org/apache/httpcomponents/httpclient/4.2/httpclient-4.2.jar $SPARK_HOME/jars
#ADD https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.11.0/jmx_prometheus_javaagent-0.11.0.jar /prometheus/
# copy files

user root

RUN mkdir -p /opt/spark/perf

COPY kmeans_clustering.py /opt/spark/perf/kmeans_clustering.py
COPY kmeans_clustering_100gb.py /opt/spark/perf/kmeans_clustering_100gb.py
COPY logistic_regression.py /opt/spark/perf/logistic_regression.py
COPY logistic_regression_25GB.py /opt/spark/perf/logistic_regression_25GB.py

COPY requirements.txt .

RUN pip install --use-pep517 --no-cache-dir -U -r requirements.txt