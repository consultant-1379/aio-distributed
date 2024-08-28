FROM armdocker.rnd.ericsson.se/sandbox/featureeng-library:v1.0

RUN apt-get update -qy && apt-get install -y python3-dev python3-pip bc


RUN python3 -m pip install --upgrade pip

COPY ad_training_featureeng-0.1.0.tar.gz ad_training_featureeng-0.1.0.tar.gz
COPY requirements.txt requirements.txt


RUN python3 -m pip install --use-pep517 --no-cache-dir -U -r requirements.txt
RUN python3 -m pip install --use-pep517 --no-cache-dir ad_training_featureeng-0.1.0.tar.gz