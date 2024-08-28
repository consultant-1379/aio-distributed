from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import boto3
import io
from io import BytesIO
import joblib
from kubernetes import client as k8sclient, config as k8sconfig
from kubernetes import watch
import threading

app = FastAPI()

client = boto3.client(
    's3',
    endpoint_url="http://<host>:<port>",
    aws_access_key_id="<accesskey>",
    aws_secret_access_key="<secret>",
)

artifact_bucket_name = 'artifacts'
feature_eng_pipeline_path = 'trained-pipeline/featureengineering/fe_trained' \
                            '.sav'
ad_thresholding_pipeline_path = \
    'trained-pipeline/AD-thresholding/thresholding_trained.sav'


class InferenceData(BaseModel):
    ndarray: List[List] = list()
    names: List[str] = list()


def get_featureeng_pipeline():
    return get_pipeline(artifact_bucket_name, feature_eng_pipeline_path,
                        'fe_trained.sav')


def get_thresholding_pipeline():
    return get_pipeline(artifact_bucket_name, ad_thresholding_pipeline_path,
                        'thresholding_trained.sav')


def read_file(bucket_name, file_path):
    try:
        obj = client.get_object(Bucket=bucket_name, Key=file_path)
        file_content = io.BytesIO(obj['Body'].read())
        return file_content

    except Exception as e:
        print("Error", e)


@app.on_event("startup")
async def startup_event():
    global featureeng_pipeline
    global thresholding_pipeline
    global forecasted_df
    print("startup - load feature engineering pipeline",flush=True)
    featureeng_pipeline = get_featureeng_pipeline()
    print("startup - load thresholding pipeline",flush=True)
    thresholding_pipeline = get_thresholding_pipeline()
    print("startup - load forecasted values from minio",flush=True)
    forecasted_values_file = read_file('data',
                                       'training/forecasted/futureweek/forecasted-values.csv')
    forecasted_df = pd.read_csv(forecasted_values_file)
    forecasted_df.columns = ['ds', 'yhat']
    forecasted_df['ds'] = pd.to_datetime(forecasted_df.ds)
    print("startup - load threshold configuration and watch for updates",flush=True)
    thread = threading.Thread(target=update_and_listen_config)
    thread.start()


@app.post("/ad/reloadconfig")
def config_reload():
    print('update config request')
    update_config()
    return "Success"


@app.post("/ad/predict")
def predict(inference_data: InferenceData):
    inputdf = pd.DataFrame(inference_data.ndarray,
                           columns=inference_data.names)
    transformed_array = featureeng_pipeline.transform(inputdf)
    transformed_df = pd.DataFrame(transformed_array,
                                  columns=['kpivalue', 'ds'])
    print('feature engineered input',transformed_df,'\n')
    transformed_df['kpivalue'] = pd.to_numeric(transformed_df["kpivalue"])
    date_list = pd.to_datetime(inputdf.ds).tolist()
    matching_forecasted_values = forecasted_df[
        forecasted_df['ds'].isin(date_list)]

    print('matching_forecasted_values\n', matching_forecasted_values, '\n')

    residual_array = transformed_df['kpivalue'].to_numpy() - \
                     matching_forecasted_values['yhat'].to_numpy()
    # print('residual_array',residual_array)
    residual_array = residual_array.reshape(-1, 1)
    response = thresholding_pipeline.predict(residual_array,
                                             threshold=threshold_value)
    response_list = [arr.tolist() for arr in response]
    index = [residual_array.tolist().index(x) for x in response_list]
    return inputdf[inputdf.index.isin(index)]


#   http://127.0.0.1:8000/ad/predict
#   uvicorn src.ad_inference_service.main:app --reload

def download_file(bucket_name, file_path, local_file_path):
    try:
        client.download_file(bucket_name, file_path, local_file_path)
    except Exception as e:
        print("Error", e)


def get_file(bucket_name, file_path):
    with BytesIO() as f:
        client.download_fileobj(Bucket=bucket_name, Key=file_path,
                                Fileobj=f)
        file_content = f.seek(0)
    return file_content


def get_pipeline(bucket_name, file_path, local_file_path):
    download_file(bucket_name,
                  file_path, local_file_path)
    pipeline_obj = joblib.load(local_file_path)
    return pipeline_obj

def update_and_listen_config():
    global threshold_value
    while True:
        try:
            k8sconfig.load_incluster_config()
            #k8sconfig.load_kube_config()
            v1 = k8sclient.CoreV1Api()
            api_response = v1.list_namespaced_config_map(namespace='adtoolbox',
                                                         label_selector='config=threshold',
                                                         timeout_seconds=10)
            threshold_str = (api_response.items[0].data).get('threshold')
            resource_version = api_response.metadata.resource_version
            threshold_value = int(threshold_str)
            print('Threshold_value ', threshold_value,'\n',flush=True)
            print('Config map resource version',
                  api_response.metadata.resource_version,
                  '\n',flush=True)
            w = watch.Watch()
            for event in w.stream(func=v1.list_namespaced_config_map,
                                  namespace='adtoolbox',
                                  label_selector='config=threshold',
                                  resource_version=resource_version):
                if event["type"] == "MODIFIED":
                    print('Received config map Modified event','\n',
                          flush=True)
                    threshold_str = event['object'].data.get('threshold')
                    threshold_value = int(threshold_str)
                    print('update thresholding value to ',threshold_value,'\n',flush=True)
                if event["type"] == "DELETED":
                    print("config map is deleted",'\n',flush=True)
                    w.stop()

        except Exception as e:
            print('exception ', e, '\n')
            pass

def update_config():
    print('updating configmap values\n')
    k8sconfig.load_incluster_config()
    v1 = k8sclient.CoreV1Api()
    api_response = v1.list_namespaced_config_map(namespace='adtoolbox',
                                                 label_selector='config=threshold',
                                                 timeout_seconds=10)
    threshold_str = (api_response.items[0].data).get('threshold')
    threshold_value = int(threshold_str)
    print('setting threshold values to ', threshold_value, '\n')