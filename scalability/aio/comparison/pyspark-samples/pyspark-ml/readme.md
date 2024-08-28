#TO build the dask image
docker build -t armdocker.rnd.ericsson.se/sandbox/sparkperf:v1.2 -f Dockerfile .

docker push armdocker.rnd.ericsson.se/sandbox/sparkperf:v1.2