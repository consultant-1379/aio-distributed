ARG RUNTIME_IMAGE=python:3.8

FROM ${RUNTIME_IMAGE}


RUN apt-get update -qy && apt-get install -y python3-dev python3-pip bc


RUN python3 -m pip install --upgrade pip

COPY ml_toolbox_client-0.1.0.tar.gz ml_toolbox_client-0.1.0.tar.gz
COPY requirements.txt requirements.txt


RUN python3 -m pip install --use-pep517 --no-cache-dir -U -r requirements.txt
RUN python3 -m pip install --use-pep517 --no-cache-dir ml_toolbox_client-0.1.0.tar.gz