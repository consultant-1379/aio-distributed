FROM gcr.io/spark-operator/spark-py:v3.0.0

USER root

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN mkdir -p /opt/spark/fpm
COPY fpm.py /opt/spark/fpm/fpm.py

RUN pip install --use-pep517 --no-cache-dir -U -r requirements.txt