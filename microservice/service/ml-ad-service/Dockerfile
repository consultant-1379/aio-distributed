
FROM armdocker.rnd.ericsson.se/sandbox/ad-library:v1.0

RUN apt-get update -qy && apt-get install -y python3-dev python3-pip bc


RUN python -m pip install --upgrade pip

COPY ml_ad_service-0.1.0.tar.gz ml_ad_service-0.1.0.tar.gz
COPY service_requirements.txt service_requirements.txt

RUN pip install --use-pep517 --no-cache-dir -U -r service_requirements.txt
RUN pip install --use-pep517 --no-cache-dir ml_ad_service-0.1.0.tar.gz


CMD ["uvicorn", "ml_ad_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
