FROM python:3.8

# Create app directory
WORKDIR /app

RUN sudo apt update
RUN sudo apt install curl
# Install app dependencies
# COPY src/requirements.txt ./

RUN pip install scikit-learn

RUN  pip install numpy==1.23.2

RUN pip install pandas==1.4.4

RUN pip install cloudpickle==2.1.0

RUN pip install lz4==4.0.0

RUN pip install dask-ml

RUN pip install boto3==1.24.14

#RUN pip install dask-kubernetes

RUN pip install "dask[complete]==2022.9.0"

#RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3

#RUN chmod 700 get_helm.sh

#RUN ./get_helm.sh

RUN pip install pyarrow
RUN pip install  s3fs

COPY kmeans_clustering.py /app
