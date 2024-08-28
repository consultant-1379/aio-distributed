FROM python:3.8

# Create app directory
WORKDIR /app

# Install app dependencies
# COPY src/requirements.txt ./

RUN pip install scikit-learn

RUN  pip install numpy==1.23.2

RUN pip install pandas==1.4.4

RUN pip install cloudpickle==2.1.0

RUN pip install lz4==4.0.0

RUN pip install dask-ml

RUN pip install boto3==1.24.14

RUN pip install "dask[complete]==2022.9.0"

RUN pip install pyarrow


COPY sklearn_dask.py /app