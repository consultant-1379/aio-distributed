
Dask helm cluster installation

helm repo add dask https://helm.dask.org 
helm repo update 
helm pull dask/dask –untar 

Dask Application & worker image
docker build -t armdocker.rnd.ericsson.se/sandbox/daskhelmcluster:v1.0 -f Dockerfile .
docker push armdocker.rnd.ericsson.se/sandbox/daskhelmcluster:v1.0


Update values.yaml worker docker image with above mentioned image armdocker.rnd.ericsson.se/sandbox/daskhelmcluster:v1.0 
 
Install helm 
helm install myhelmrelease -f dask/values.yaml ./dask -n daskhelm 