
#TO build the dask image
docker build -t armdocker.rnd.ericsson.se/sandbox/sklearndask:v1.0 -f Dockerfile . 

docker push armdocker.rnd.ericsson.se/sandbox/sklearndask:v1.0