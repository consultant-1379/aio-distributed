FROM armdocker.rnd.ericsson.se/sandbox/ad-library:v1.0

RUN apt-get update -qy && apt-get install -y python3-dev python3-pip bc

RUN python -m pip install --upgrade pip

COPY ad_inference_service-0.1.0.tar.gz ad_inference_service-0.1.0.tar.gz
COPY requirements.txt requirements.txt

RUN pip install --use-pep517 --no-cache-dir -U -r requirements.txt
RUN pip install --use-pep517 --no-cache-dir ad_inference_service-0.1.0.tar.gz


CMD ["uvicorn", "ad_inference_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
