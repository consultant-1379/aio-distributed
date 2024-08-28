ARG RUNTIME_IMAGE=python:3.8

FROM ${RUNTIME_IMAGE}

RUN apt-get update -qy && apt-get install -y python3-dev python3-pip bc


RUN python -m pip install --upgrade pip

COPY ml_preprocessing_lib-0.1.0.tar.gz ml_preprocessing_lib-0.1.0.tar.gz
COPY lib_requirements.txt lib_requirements.txt

  
RUN pip install --use-pep517 --no-cache-dir -U -r lib_requirements.txt
RUN pip install --use-pep517 --no-cache-dir ml_preprocessing_lib-0.1.0.tar.gz