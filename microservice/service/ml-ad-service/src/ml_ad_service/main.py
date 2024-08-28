from typing import Union, List

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ml_ad_lib.base.anomaly_detection import create_ad_pipeline
import os
import shutil
import joblib

app = FastAPI()


class ADEstimatorParameters(BaseModel):
    key: str
    value: str


class ADEstimatorStep(BaseModel):
    step: str
    params: Union[List[ADEstimatorParameters], None] = None


def delete_directory(folder_path: str):
    shutil.rmtree(folder_path)


def create_directory(dir_path):
    os.makedirs(dir_path)


def create_pipeline_input(estimator_step: ADEstimatorStep):
    transformer = dict()
    transformer['step'] = estimator_step.step
    if estimator_step.params:
        for individual_param in estimator_step.params:
            transformer[individual_param.key] = individual_param.value
    return transformer


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta



@app.post("/ad/createpipeline", response_class=FileResponse)
def create_ad_forecasting_pipeline(estimatorg_step: ADEstimatorStep,
                                   background_tasks: BackgroundTasks):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    transformer_pipeline_steps = create_pipeline_input(
        estimatorg_step)
    ad_pipeline = create_ad_pipeline(
        transformer_pipeline_steps)
    pipeline_id = str(id(ad_pipeline))
    dir_path = ROOT_DIR + '/' + pipeline_id
    file_path = dir_path + "/ad.sav"
    create_directory(dir_path)
    joblib.dump(ad_pipeline, file_path)
    background_tasks.add_task(delete_directory, dir_path)
    return file_path

#   http://127.0.0.1:8000/ad/createpipeline
# uvicorn src.ml_ad_service.main:app --reload
